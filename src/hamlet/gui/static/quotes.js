// A line of Shakespeare while the machine thinks.
//
// The runs behind this interface are long: one simulated chain is around a
// minute, a dataset is hours, and the only thing the page could offer for that
// wait was "142 of 3000 chains". The package is called HamLeT, so the wait
// quotes the play.
//
// Every line here is Shakespeare and therefore public domain. They are chosen
// for the situation rather than drawn at random from the collected works --
// waiting, doubt, measurement, and the patience a long run asks for -- and a
// few are held back until the wait has earned them.

// How long one line stays put. The jobs list re-renders every three seconds,
// so the quote has to be a function of the clock rather than of the render, or
// it would flicker through the whole bank while you read it.
const QUOTE_ROTATE_SECONDS = 30;

// `after` withholds a line until the wait is that many seconds old.
const SHAKESPEARE_QUOTES = [
  { line: "Season your admiration for a while.", source: "Hamlet, I.ii" },
  { line: "Though this be madness, yet there is method in&rsquo;t.", source: "Hamlet, II.ii" },
  {
    line: "There are more things in heaven and earth, Horatio, than are dreamt "
      + "of in your philosophy.",
    source: "Hamlet, I.v",
  },
  { line: "Doubt thou the stars are fire; doubt that the sun doth move.", source: "Hamlet, II.ii" },
  { line: "The readiness is all.", source: "Hamlet, V.ii" },
  { line: "By indirections find directions out.", source: "Hamlet, II.i" },
  { line: "We know what we are, but know not what we may be.", source: "Hamlet, IV.v" },
  {
    line: "I could be bounded in a nutshell and count myself a king of "
      + "infinite space.",
    source: "Hamlet, II.ii",
  },
  { line: "Give every man thy ear, but few thy voice.", source: "Hamlet, I.iii" },
  { line: "The play&rsquo;s the thing.", source: "Hamlet, II.ii" },
  { line: "Brevity is the soul of wit.", source: "Hamlet, II.ii" },
  { line: "Let be.", source: "Hamlet, V.ii" },
  { line: "Our doubts are traitors.", source: "Measure for Measure, I.iv" },
  { line: "Truth is truth to the end of reckoning.", source: "Measure for Measure, V.i" },
  { line: "Wisely and slow; they stumble that run fast.", source: "Romeo and Juliet, II.iii" },
  { line: "All things are ready, if our minds be so.", source: "Henry V, IV.iii" },
  { line: "Men of few words are the best men.", source: "Henry V, III.ii" },
  { line: "How far that little candle throws his beams!", source: "The Merchant of Venice, V.i" },
  { line: "I am not bound to please thee with my answers.", source: "The Merchant of Venice, IV.i" },
  { line: "What&rsquo;s past is prologue.", source: "The Tempest, II.i" },
  { line: "We are such stuff as dreams are made on.", source: "The Tempest, IV.i" },
  { line: "Journeys end in lovers meeting.", source: "Twelfth Night, II.iii" },
  { line: "Come what come may, time and the hour runs through the roughest day.", source: "Macbeth, I.iii" },
  { line: "Nothing will come of nothing.", source: "King Lear, I.i" },
  { line: "Ripeness is all.", source: "King Lear, V.ii" },

  // Three minutes in, the wait is long enough to be joked about.
  { line: "The time is out of joint.", source: "Hamlet, I.v", after: 180 },
  { line: "Rest, rest, perturbed spirit!", source: "Hamlet, I.v", after: 180 },
  { line: "Ay, there&rsquo;s the rub.", source: "Hamlet, III.i", after: 180 },
  { line: "Words, words, words.", source: "Hamlet, II.ii", after: 180 },
  { line: "How poor are they that have not patience!", source: "Othello, II.iii", after: 180 },
  { line: "Time travels in divers paces with divers persons.", source: "As You Like It, III.ii", after: 180 },
  {
    line: "Sit by my side, and let the world slip: we shall ne&rsquo;er be younger.",
    source: "The Taming of the Shrew, Induction ii",
    after: 180,
  },

  // A quarter of an hour. Anyone still here is generating a dataset.
  { line: "How long hast thou been a grave-maker?", source: "Hamlet, V.i", after: 900 },
  { line: "I wasted time, and now doth time waste me.", source: "Richard II, V.v", after: 900 },
  {
    line: "Tomorrow, and tomorrow, and tomorrow, creeps in this petty pace "
      + "from day to day.",
    source: "Macbeth, V.v",
    after: 900,
  },
  { line: "O, call back yesterday, bid time return.", source: "Richard II, III.ii", after: 900 },
  { line: "Now is the winter of our discontent.", source: "Richard III, I.i", after: 900 },
];

// The line each wait is currently showing, keyed by its seed.
//
// This exists because the surfaces do not share a clock. The jobs list polls
// every three seconds and a job's own page every two, so deriving the line
// from the elapsed time sampled at render meant the two disagreed whenever
// their polls fell either side of a rotation -- the same run quoting two
// different lines on two pages at once. Holding the choice here makes every
// surface read one answer, and rotation a decision taken on a timestamp
// rather than a side effect of when something happened to redraw.
const showing = new Map();

// A small stable spread over the seed, so two jobs waiting at the same moment
// are not handed the same line.
function quoteSeed(seed) {
  let hash = 0;
  for (const character of String(seed)) {
    hash = (hash * 31 + character.codePointAt(0)) % 100003;
  }
  return hash;
}

/** One quote for a running job. There is always one.
 *
 * `seed` distinguishes concurrent waits; `elapsedSeconds` advances the line,
 * so a long run reads several rather than staring at one, and unlocks the
 * tiers reserved for a wait that has earned them.
 */
function waitingQuote(seed, elapsedSeconds) {
  const elapsed = Number(elapsedSeconds) || 0;
  const pool = SHAKESPEARE_QUOTES.filter((q) => elapsed >= (q.after || 0));
  const key = String(seed);
  const now = Date.now();
  let state = showing.get(key);
  // A new wait, one whose line has had its thirty seconds, or one that has
  // just crossed into a tier with lines it could not draw before.
  if (!state || now - state.since >= QUOTE_ROTATE_SECONDS * 1000
      || state.size !== pool.length) {
    const previous = state ? state.index : quoteSeed(key) % pool.length;
    state = {
      // Stepping rather than re-randomising: it cannot repeat the line that
      // is already on screen, which random choice does about one time in
      // thirty and which reads as the rotation having broken.
      index: (previous + (state ? 1 : 0)) % pool.length,
      since: now,
      size: pool.length,
    };
    showing.set(key, state);
  }
  const quote = pool[state.index % pool.length];
  // The text is this file's own, so it is markup rather than escaped input --
  // that is what lets a line carry its own curly apostrophe.
  return `<p class="wait-quote">
    <span class="wait-quote-line">&ldquo;${quote.line}&rdquo;</span>
    <span class="wait-quote-source">&mdash; ${quote.source}</span>
  </p>`;
}
