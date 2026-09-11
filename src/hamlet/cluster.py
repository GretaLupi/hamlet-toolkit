"""Running the heavy stages on somebody else's cluster.

Generation is hours and training a large model is not much better, so the work
belongs on a cluster. The problem is that there is no such thing as "a
cluster": every group has a different scheduler, different queue names,
different module system, and a different idea of how to ask for a GPU. Writing
support for Slurm would help the groups that have Slurm and nobody else.

So a scheduler here is **data, not code**. A :class:`SchedulerProfile` says
which command submits, what a directive line looks like, and how to read a job
id out of the reply; the built-in profiles cover Slurm, PBS/Torque, LSF, SGE
and "no scheduler at all", and an unknown one is a YAML file the user writes
rather than a patch to this package.

Reaching the cluster is likewise deliberately unclever: everything goes through
the system ``ssh`` and ``rsync`` binaries. That means the user's existing
config, keys, agent, jump hosts and kerberos all work exactly as they already
do, and this package never handles a credential. A cluster you can ``ssh`` to
is a cluster this can drive.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
import re
import shlex
import subprocess
from typing import Any, Mapping, Sequence

CLUSTER_CONFIG_SCHEMA_VERSION = 1

# The resource names this package speaks. Each profile maps them onto whatever
# its scheduler calls them, so a project configuration never has to.
# Passwordless access is a precondition, so it is spelled into the default
# command rather than left to whatever the user's ssh config happens to do.
DEFAULT_SSH_COMMAND = ("ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10")

RESOURCE_FIELDS = (
    "job_name",
    "nodes",
    "cpus",
    "gpus",
    "memory",
    "walltime",
    "queue",
    "account",
    "stdout",
    "stderr",
)


@dataclass(frozen=True)
class SchedulerProfile:
    """How one batch system is spoken to.

    ``directives`` maps a resource name from :data:`RESOURCE_FIELDS` onto the
    option text for this scheduler, with ``{value}`` standing for what the user
    asked for. A resource the profile has no entry for is simply not emitted,
    which is how a scheduler that cannot express something stays usable rather
    than becoming an error.
    """

    name: str
    submit_command: tuple[str, ...]
    directive_prefix: str
    directives: dict[str, str] = field(default_factory=dict)
    status_command: tuple[str, ...] = ()
    cancel_command: tuple[str, ...] = ()
    job_id_pattern: str = r"(\d+)"
    submit_takes_script_on_stdin: bool = False
    notes: str = ""

    def directive_lines(self, resources: Mapping[str, Any]) -> list[str]:
        lines = []
        for name in RESOURCE_FIELDS:
            value = resources.get(name)
            if value in (None, "", 0):
                continue
            template = self.directives.get(name)
            if template is None:
                continue
            lines.append(f"{self.directive_prefix} {template.format(value=value)}")
        return lines

    def parse_job_id(self, submit_output: str) -> str:
        """Pull the job id out of whatever the submit command printed.

        Falls back to the first non-empty line: several schedulers print the id
        and nothing else, and a job that ran but whose id could not be parsed is
        better reported with a best guess than lost.
        """
        match = re.search(self.job_id_pattern, submit_output)
        if match:
            return match.group(match.lastindex or 0)
        for line in submit_output.splitlines():
            if line.strip():
                return line.strip()
        return ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "submit_command": list(self.submit_command),
            "directive_prefix": self.directive_prefix,
            "directives": dict(self.directives),
            "status_command": list(self.status_command),
            "cancel_command": list(self.cancel_command),
            "job_id_pattern": self.job_id_pattern,
            "submit_takes_script_on_stdin": self.submit_takes_script_on_stdin,
            "notes": self.notes,
        }


BUILT_IN_PROFILES: dict[str, SchedulerProfile] = {
    "slurm": SchedulerProfile(
        name="slurm",
        submit_command=("sbatch",),
        directive_prefix="#SBATCH",
        directives={
            "job_name": "--job-name={value}",
            "nodes": "--nodes={value}",
            "cpus": "--cpus-per-task={value}",
            "gpus": "--gres=gpu:{value}",
            "memory": "--mem={value}",
            "walltime": "--time={value}",
            "queue": "--partition={value}",
            "account": "--account={value}",
            "stdout": "--output={value}",
            "stderr": "--error={value}",
        },
        status_command=("squeue", "-j"),
        cancel_command=("scancel",),
        job_id_pattern=r"Submitted batch job (\d+)",
        notes="The most common scheduler in academic HPC.",
    ),
    "pbs": SchedulerProfile(
        name="pbs",
        submit_command=("qsub",),
        directive_prefix="#PBS",
        directives={
            "job_name": "-N {value}",
            "cpus": "-l ncpus={value}",
            "gpus": "-l ngpus={value}",
            "memory": "-l mem={value}",
            "walltime": "-l walltime={value}",
            "queue": "-q {value}",
            "account": "-A {value}",
            "stdout": "-o {value}",
            "stderr": "-e {value}",
        },
        status_command=("qstat",),
        cancel_command=("qdel",),
        job_id_pattern=r"^(\S+)",
        notes=(
            "PBS Pro and Torque. Sites differ on whether resources go in one "
            "`-l select=` line; if yours does, copy this profile and edit it."
        ),
    ),
    "lsf": SchedulerProfile(
        name="lsf",
        submit_command=("bsub",),
        directive_prefix="#BSUB",
        directives={
            "job_name": "-J {value}",
            "cpus": "-n {value}",
            "gpus": '-gpu "num={value}"',
            "memory": '-R "rusage[mem={value}]"',
            "walltime": "-W {value}",
            "queue": "-q {value}",
            "account": "-P {value}",
            "stdout": "-o {value}",
            "stderr": "-e {value}",
        },
        status_command=("bjobs",),
        cancel_command=("bkill",),
        job_id_pattern=r"Job <(\d+)>",
        submit_takes_script_on_stdin=True,
        notes="IBM Spectrum LSF. bsub reads the script from standard input.",
    ),
    "sge": SchedulerProfile(
        name="sge",
        submit_command=("qsub",),
        directive_prefix="#$",
        directives={
            "job_name": "-N {value}",
            "cpus": "-pe smp {value}",
            "memory": "-l h_vmem={value}",
            "walltime": "-l h_rt={value}",
            "queue": "-q {value}",
            "account": "-A {value}",
            "stdout": "-o {value}",
            "stderr": "-e {value}",
        },
        status_command=("qstat", "-j"),
        cancel_command=("qdel",),
        job_id_pattern=r"Your job (\d+)",
        notes=(
            "Sun/Son of Grid Engine. GPUs are requested through a site-specific "
            "complex, so add one to `directives` if yours needs it."
        ),
    ),
    "none": SchedulerProfile(
        name="none",
        submit_command=("nohup",),
        directive_prefix="#",
        directives={},
        status_command=("ps", "-p"),
        cancel_command=("kill",),
        job_id_pattern=r"(\d+)",
        notes=(
            "No batch system: the job is started in the background on the "
            "machine itself. For a workstation you can ssh to, or a login node "
            "you are allowed to compute on."
        ),
    ),
}


def available_profiles() -> dict[str, dict[str, Any]]:
    """Every built-in scheduler, for a form or a `--help` listing."""
    return {name: profile.to_dict() for name, profile in BUILT_IN_PROFILES.items()}


def get_profile(scheduler: str | SchedulerProfile) -> SchedulerProfile:
    if isinstance(scheduler, SchedulerProfile):
        return scheduler
    try:
        return BUILT_IN_PROFILES[str(scheduler)]
    except KeyError:
        raise ValueError(
            f"unknown scheduler {scheduler!r}; built-in profiles are "
            f"{sorted(BUILT_IN_PROFILES)}. If yours is not one of these, write "
            "a `scheduler:` block in the cluster configuration instead of "
            "naming one."
        ) from None


def profile_from_mapping(payload: Mapping[str, Any]) -> SchedulerProfile:
    """Build a profile from a user's own description of their scheduler.

    This is the escape hatch that makes the feature general: a site with a
    scheduler nobody here has heard of writes it down once instead of waiting
    for support to be added.
    """
    known = {
        "name", "submit_command", "directive_prefix", "directives",
        "status_command", "cancel_command", "job_id_pattern",
        "submit_takes_script_on_stdin", "notes", "extends",
    }
    unknown = set(payload) - known
    if unknown:
        raise ValueError(f"scheduler has unknown fields: {sorted(unknown)}")

    base = (
        get_profile(str(payload["extends"]))
        if payload.get("extends")
        else SchedulerProfile(name="custom", submit_command=(), directive_prefix="#")
    )
    directives = dict(base.directives)
    directives.update(payload.get("directives") or {})
    for name, template in directives.items():
        if name not in RESOURCE_FIELDS:
            raise ValueError(
                f"scheduler directive {name!r} is not one of {list(RESOURCE_FIELDS)}"
            )
        if "{value}" not in template:
            raise ValueError(
                f"scheduler directive {name!r} must contain {{value}}: got {template!r}"
            )

    def command(key: str, fallback: tuple[str, ...]) -> tuple[str, ...]:
        given = payload.get(key)
        if given is None:
            return fallback
        if isinstance(given, str):
            return tuple(shlex.split(given))
        return tuple(str(item) for item in given)

    profile = replace(
        base,
        name=str(payload.get("name", base.name or "custom")),
        submit_command=command("submit_command", base.submit_command),
        directive_prefix=str(payload.get("directive_prefix", base.directive_prefix)),
        directives=directives,
        status_command=command("status_command", base.status_command),
        cancel_command=command("cancel_command", base.cancel_command),
        job_id_pattern=str(payload.get("job_id_pattern", base.job_id_pattern)),
        submit_takes_script_on_stdin=bool(
            payload.get("submit_takes_script_on_stdin",
                        base.submit_takes_script_on_stdin)
        ),
        notes=str(payload.get("notes", base.notes)),
    )
    if not profile.submit_command:
        raise ValueError("a scheduler needs a submit_command")
    try:
        re.compile(profile.job_id_pattern)
    except re.error as exc:
        raise ValueError(f"job_id_pattern is not a valid regex: {exc}") from exc
    return profile


def _looks_like_an_auth_refusal(detail: str) -> bool:
    """Whether ssh declined for want of a usable key.

    Worth telling apart from an unreachable host: one is fixed with
    `ssh-copy-id` and the other with a VPN or a corrected address, and the
    generic "could not reach it" sends people to check the wrong one. Under
    BatchMode ssh says so in a small number of recognisable ways.
    """
    lowered = detail.lower()
    return any(
        phrase in lowered
        for phrase in (
            "permission denied",
            "batch mode",
            "publickey",
            "no supported authentication",
            "host key verification failed",
        )
    )


def _connection_hint(reachable: bool, needs_key: bool, host: str | None) -> str | None:
    if reachable:
        return None
    target = host or "<host>"
    if needs_key:
        return (
            "ssh reached it but would not log in without a password, and this "
            "package never types one -- runs start on a background thread with "
            "no terminal to prompt at. Set up key-based access once:\n"
            f"    ssh-keygen -t ed25519        # if you have no key yet\n"
            f"    ssh-copy-id {target}\n"
            f"    ssh {target} true            # must succeed without asking\n"
            "If your key has a passphrase, load it into ssh-agent first."
        )
    return (
        "ssh could not reach it. This package uses your own ssh, so check that "
        f"`ssh {target}` works in a terminal first -- including any jump host, "
        "VPN, or key your ~/.ssh/config sets up."
    )


@dataclass(frozen=True)
class ClusterConfig:
    """Where a run should go, and what it should ask for when it gets there.

    ``host`` of ``None`` means "submit here": on a login node you are already
    sitting on, there is nothing to ssh to and staging would be a copy of a
    directory onto itself.
    """

    remote_dir: str
    scheduler: SchedulerProfile
    host: str | None = None
    resources: dict[str, Any] = field(default_factory=dict)
    setup_lines: tuple[str, ...] = ()
    python: str = "python"
    # BatchMode refuses to ask for anything. Every cluster operation here runs
    # on a worker thread with no terminal attached, so a password or passphrase
    # prompt has nobody to answer it: ssh would wait for an input that can
    # never arrive and the job would hang until the interface was killed.
    # Failing immediately turns that into a message that names the fix.
    # Key-based access is therefore a requirement, not a preference.
    ssh_command: tuple[str, ...] = DEFAULT_SSH_COMMAND
    rsync_command: tuple[str, ...] = ("rsync", "-az", "--delete")
    schema_version: int = CLUSTER_CONFIG_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CLUSTER_CONFIG_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported cluster schema version {self.schema_version}; "
                f"this build reads {CLUSTER_CONFIG_SCHEMA_VERSION}"
            )
        if not str(self.remote_dir).strip():
            raise ValueError("remote_dir cannot be empty")
        unknown = set(self.resources) - set(RESOURCE_FIELDS)
        if unknown:
            raise ValueError(
                f"unknown resource(s) {sorted(unknown)}; this package speaks "
                f"{list(RESOURCE_FIELDS)}. Anything else your site needs goes "
                "in the scheduler's own `directives`."
            )

    @property
    def is_remote(self) -> bool:
        return bool(self.host)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ClusterConfig":
        known = {
            "cluster_schema_version", "host", "remote_dir", "scheduler",
            "resources", "setup", "python", "ssh_command", "rsync_command",
        }
        unknown = set(payload) - known
        if unknown:
            raise ValueError(f"cluster configuration has unknown fields: {sorted(unknown)}")
        scheduler = payload.get("scheduler", "slurm")
        profile = (
            profile_from_mapping(scheduler)
            if isinstance(scheduler, Mapping)
            else get_profile(str(scheduler))
        )
        setup = payload.get("setup") or ()
        if isinstance(setup, str):
            setup = [line for line in setup.splitlines() if line.strip()]

        def command(key: str, fallback: tuple[str, ...]) -> tuple[str, ...]:
            given = payload.get(key)
            if given is None:
                return fallback
            if isinstance(given, str):
                return tuple(shlex.split(given))
            return tuple(str(item) for item in given)

        if "remote_dir" not in payload:
            raise ValueError("cluster configuration requires remote_dir")
        return cls(
            remote_dir=str(payload["remote_dir"]),
            scheduler=profile,
            host=str(payload["host"]) if payload.get("host") else None,
            resources=dict(payload.get("resources") or {}),
            setup_lines=tuple(str(line) for line in setup),
            python=str(payload.get("python", "python")),
            ssh_command=command("ssh_command", DEFAULT_SSH_COMMAND),
            rsync_command=command("rsync_command", ("rsync", "-az", "--delete")),
            schema_version=int(
                payload.get("cluster_schema_version", CLUSTER_CONFIG_SCHEMA_VERSION)
            ),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> "ClusterConfig":
        import json

        config_path = Path(path).expanduser().resolve()
        text = config_path.read_text(encoding="utf-8")
        if config_path.suffix.lower() == ".json":
            payload = json.loads(text)
        else:
            try:
                import yaml
            except ImportError as exc:  # pragma: no cover
                raise ImportError("YAML configuration requires PyYAML") from exc
            payload = yaml.safe_load(text)
        if not isinstance(payload, Mapping):
            raise ValueError("cluster configuration must contain a mapping")
        return cls.from_mapping(payload)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_schema_version": self.schema_version,
            "host": self.host,
            "remote_dir": self.remote_dir,
            "scheduler": self.scheduler.name,
            "resources": dict(self.resources),
            "setup": list(self.setup_lines),
            "python": self.python,
        }


def render_job_script(
    cluster: ClusterConfig,
    command: str,
    *,
    job_name: str = "hamlet",
    work_dir: str | None = None,
) -> str:
    """The batch script that runs one HamLeT command on the cluster.

    Written to be read: someone whose site needs one more directive should be
    able to open this file, add the line, and submit it by hand, rather than
    having to come back here.
    """
    profile = cluster.scheduler
    resources = {"job_name": job_name, **cluster.resources}
    resources.setdefault("stdout", f"{job_name}-%j.out" if profile.name == "slurm"
                         else f"{job_name}.out")
    directory = work_dir or cluster.remote_dir

    lines = ["#!/bin/bash"]
    directives = profile.directive_lines(resources)
    if directives:
        lines.extend(directives)
    lines += [
        "",
        f"# Generated by HamLeT for the {profile.name} scheduler.",
        "# Edit and submit by hand if your site needs anything else.",
        "",
        # A batch job that keeps going after a failed step reports success for
        # a run that produced nothing, which is worse than failing.
        "set -euo pipefail",
        "",
        f"cd {shlex.quote(directory)}",
    ]
    if cluster.setup_lines:
        lines += ["", "# Environment, as configured for this cluster."]
        lines += list(cluster.setup_lines)
    lines += [
        "",
        "echo \"host: $(hostname)\"",
        "echo \"started: $(date -Is)\"",
        "",
        command,
        "",
        "echo \"finished: $(date -Is)\"",
    ]
    return "\n".join(lines) + "\n"


def project_command(
    cluster: ClusterConfig, config_name: str = "project.yaml", *, dry_run: bool = False
) -> str:
    """The command a batch script should run for a project configuration."""
    parts = [cluster.python, "-m", "hamlet.project_cli", "run", config_name]
    if dry_run:
        parts.append("--dry-run")
    return " ".join(shlex.quote(part) for part in parts)


# --- talking to the cluster --------------------------------------------------

@dataclass(frozen=True)
class CommandResult:
    argv: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    def raise_for_status(self, what: str) -> "CommandResult":
        if not self.ok:
            raise RuntimeError(
                f"{what} failed ({self.returncode}): "
                f"{(self.stderr or self.stdout).strip()}\n"
                f"command: {' '.join(self.argv)}"
            )
        return self


class LocalRunner:
    """Runs commands with :mod:`subprocess`.

    Separated behind a tiny interface so the whole submission path can be
    tested without a cluster: a test injects a recorder and asserts on the
    argv, which is where the mistakes in this module actually live.
    """

    def run(
        self, argv: Sequence[str], *, input_text: str | None = None, timeout: float = 120.0
    ) -> CommandResult:
        completed = subprocess.run(
            list(argv),
            input=input_text,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return CommandResult(
            argv=tuple(argv),
            returncode=completed.returncode,
            stdout=completed.stdout or "",
            stderr=completed.stderr or "",
        )


class ClusterSession:
    """Everything this package does to a cluster, in one place."""

    def __init__(self, cluster: ClusterConfig, runner: Any | None = None) -> None:
        self.cluster = cluster
        self.runner = runner or LocalRunner()

    # -- shaping commands --
    def remote_argv(self, command: str) -> tuple[str, ...]:
        """Wrap a shell command so it runs on the cluster.

        Local when there is no host: on a login node the "remote" is here, and
        going through ssh to localhost would only add a way to fail.
        """
        if not self.cluster.is_remote:
            return ("bash", "-lc", command)
        return (*self.cluster.ssh_command, str(self.cluster.host), command)

    def run_remote(self, command: str, *, input_text: str | None = None) -> CommandResult:
        return self.runner.run(self.remote_argv(command), input_text=input_text)

    # -- operations --
    def check_connection(self) -> dict[str, Any]:
        """Confirm the cluster is reachable and the scheduler is actually there.

        Both halves matter: an ssh that works but a `sbatch` that does not is a
        configuration mistake worth finding now rather than at submit time.
        """
        submit = self.cluster.scheduler.submit_command[0]
        result = self.run_remote(
            f"echo HAMLET_OK; command -v {shlex.quote(submit)} || echo NO_SCHEDULER"
        )
        reachable = result.ok and "HAMLET_OK" in result.stdout
        scheduler_found = reachable and "NO_SCHEDULER" not in result.stdout
        detail = (result.stderr or result.stdout).strip()
        needs_key = not reachable and _looks_like_an_auth_refusal(detail)
        return {
            "host": self.cluster.host or "this machine",
            "reachable": reachable,
            "scheduler": self.cluster.scheduler.name,
            "scheduler_found": scheduler_found,
            "detail": detail,
            "needs_key": needs_key,
            "hint": _connection_hint(reachable, needs_key, self.cluster.host),
        }

    def stage(self, local_dir: str | Path) -> CommandResult:
        """Copy a project directory to the cluster.

        rsync because a resumed generation run means copying a directory that
        mostly already exists, and because it is on every cluster.
        """
        source = f"{Path(local_dir).resolve()}/"
        if not self.cluster.is_remote:
            destination = self.cluster.remote_dir
            if Path(destination).resolve() == Path(source).resolve():
                return CommandResult(("true",), 0, "already in place", "")
        else:
            destination = f"{self.cluster.host}:{self.cluster.remote_dir}"
        self.run_remote(f"mkdir -p {shlex.quote(self.cluster.remote_dir)}")
        return self.runner.run(
            (*self.cluster.rsync_command, source, destination), timeout=3600.0
        )

    def submit(self, script_text: str, *, script_name: str = "hamlet-job.sh") -> dict[str, Any]:
        """Write the script on the cluster and submit it."""
        profile = self.cluster.scheduler
        remote_script = f"{self.cluster.remote_dir.rstrip('/')}/{script_name}"
        write = self.run_remote(
            f"mkdir -p {shlex.quote(self.cluster.remote_dir)} && "
            f"cat > {shlex.quote(remote_script)} && chmod +x {shlex.quote(remote_script)}",
            input_text=script_text,
        )
        write.raise_for_status("writing the job script")

        submit = " ".join(shlex.quote(part) for part in profile.submit_command)
        if profile.name == "none":
            # No scheduler: start it detached and report the pid, so the job
            # survives the ssh connection closing.
            command = (
                f"cd {shlex.quote(self.cluster.remote_dir)} && "
                f"nohup bash {shlex.quote(script_name)} "
                f"> {shlex.quote(script_name)}.out 2>&1 & echo $!"
            )
        elif profile.submit_takes_script_on_stdin:
            command = (
                f"cd {shlex.quote(self.cluster.remote_dir)} && "
                f"{submit} < {shlex.quote(script_name)}"
            )
        else:
            command = (
                f"cd {shlex.quote(self.cluster.remote_dir)} && "
                f"{submit} {shlex.quote(script_name)}"
            )
        result = self.run_remote(command)
        result.raise_for_status("submitting the job")
        job_id = profile.parse_job_id(result.stdout)
        return {
            "job_id": job_id,
            "scheduler": profile.name,
            "script": remote_script,
            "output": result.stdout.strip(),
            "remote_dir": self.cluster.remote_dir,
        }

    def status(self, job_id: str) -> dict[str, Any]:
        profile = self.cluster.scheduler
        if not profile.status_command:
            return {"job_id": job_id, "known": False,
                    "detail": f"the {profile.name} profile has no status command"}
        command = " ".join(
            shlex.quote(part) for part in (*profile.status_command, job_id)
        )
        result = self.run_remote(command)
        return {
            "job_id": job_id,
            # A scheduler drops a finished job from its queue, so "not listed"
            # usually means done rather than missing -- said plainly, because
            # the opposite reading would look like a lost job.
            "known": result.ok and job_id in result.stdout,
            "detail": (result.stdout or result.stderr).strip(),
            "note": (
                "Most schedulers stop listing a job once it finishes, so an "
                "empty answer usually means it is done. Fetch the results to "
                "find out."
            ),
        }

    def cancel(self, job_id: str) -> dict[str, Any]:
        profile = self.cluster.scheduler
        if not profile.cancel_command:
            raise RuntimeError(f"the {profile.name} profile has no cancel command")
        command = " ".join(
            shlex.quote(part) for part in (*profile.cancel_command, job_id)
        )
        result = self.run_remote(command)
        return {
            "job_id": job_id,
            "cancelled": result.ok,
            "detail": (result.stdout or result.stderr).strip(),
        }

    def fetch(self, local_dir: str | Path) -> CommandResult:
        """Bring the results back.

        Deliberately not ``--delete``: the local directory is the one the user
        has been working in, and a fetch is not a reason to remove anything
        from it.
        """
        destination = f"{Path(local_dir).resolve()}/"
        source = (
            f"{self.cluster.host}:{self.cluster.remote_dir}/"
            if self.cluster.is_remote
            else f"{self.cluster.remote_dir.rstrip('/')}/"
        )
        pull = [part for part in self.cluster.rsync_command if part != "--delete"]
        return self.runner.run((*pull, source, destination), timeout=3600.0)


def make_portable(config_path: str | Path) -> dict[str, Any]:
    """Rewrite a project configuration's paths to be relative to its own file.

    A configuration the guided form wrote records absolute local paths, which
    name nothing on a cluster. ``ProjectConfig`` resolves relative paths against
    the configuration's own directory, so a directory whose paths are all
    relative runs wherever it is put -- which is what makes staging a copy
    rather than a rewrite.
    """
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ImportError("rewriting a configuration requires PyYAML") from exc

    path = Path(config_path).expanduser().resolve()
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("project configuration must contain a mapping")
    base = path.parent
    rewritten: list[str] = []
    outside: list[str] = []

    def relative(value: Any) -> Any:
        if not isinstance(value, str) or not value.startswith("/"):
            return value
        candidate = Path(value)
        try:
            made = candidate.relative_to(base)
        except ValueError:
            # A path outside the project directory cannot be made portable by
            # rewriting it, so it is reported rather than mangled: staging will
            # not carry it, and the user has to decide what to do.
            outside.append(value)
            return value
        rewritten.append(value)
        return str(made)

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            return {key: walk(item) for key, item in node.items()}
        if isinstance(node, list):
            return [walk(item) for item in node]
        return relative(node)

    portable = walk(payload)
    path.write_text(yaml.safe_dump(portable, sort_keys=False), encoding="utf-8")
    return {
        "config_path": str(path),
        "project_dir": str(base),
        "rewritten": rewritten,
        "outside_project_dir": outside,
        "portable": not outside,
    }


EXAMPLE_CLUSTER_CONFIG = """\
# Where HamLeT should send heavy runs. Everything goes through your own ssh, so
# whatever `ssh <host>` already does -- keys, agent, jump hosts -- keeps working
# and this package never sees a credential.
#
# Key-based access is required, not merely convenient: runs are submitted from
# a background thread with no terminal, so nothing can answer a password
# prompt. `ssh <host> true` must succeed without asking. If it does not:
#     ssh-keygen -t ed25519        # only if you have no key yet
#     ssh-copy-id user@cluster.example.edu
cluster_schema_version: 1

# Leave `host` out entirely if you are already on the login node.
host: user@cluster.example.edu
remote_dir: /scratch/user/hamlet-runs/my-chain

# One of: slurm, pbs, lsf, sge, none. If your site runs something else, replace
# this line with the `scheduler:` block at the bottom of this file.
scheduler: slurm

resources:
  cpus: 8
  # Generation is CPU-bound DMRG, so a GPU only helps the training stage, and
  # only for the keras models. Ask for one when you are training a network.
  gpus: 0
  memory: 16G
  walltime: "24:00:00"
  queue: batch
  # account: your-project

# Run before the job. Whatever your site needs to make `python -m hamlet` work.
setup:
  - module load python/3.11
  - source ~/venvs/hamlet/bin/activate

python: python

# --- for a scheduler that is not one of the built-in ones --------------------
# Replace the `scheduler: slurm` line above with a block like this. `extends`
# starts from a built-in profile and changes only what differs at your site.
#
# scheduler:
#   name: our-scheduler
#   extends: pbs
#   submit_command: [qsub, -V]
#   directive_prefix: "#PBS"
#   directives:
#     cpus: "-l select=1:ncpus={value}"
#     gpus: "-l select=1:ngpus={value}"
#   job_id_pattern: "^(\\\\d+)"
"""


__all__ = [
    "BUILT_IN_PROFILES",
    "CLUSTER_CONFIG_SCHEMA_VERSION",
    "ClusterConfig",
    "ClusterSession",
    "CommandResult",
    "EXAMPLE_CLUSTER_CONFIG",
    "LocalRunner",
    "RESOURCE_FIELDS",
    "SchedulerProfile",
    "available_profiles",
    "get_profile",
    "make_portable",
    "profile_from_mapping",
    "project_command",
    "render_job_script",
]
