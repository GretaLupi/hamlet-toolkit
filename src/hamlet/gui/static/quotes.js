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
//
// Provenance: the wording follows Project Gutenberg's Complete Works (ebook
// 100), and every act and scene was checked against a second edition (the MIT
// text) rather than written from memory -- which caught three lines that were
// misquoted here before. Where the two editions differ (`sleave` or `sleeve`,
// `every thing` or `everything`) the Gutenberg reading is kept, so there is
// one answer to where a line came from. Scene divisions are edition-dependent
// in a few plays; the conventional numbering is used.

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
  { line: "Give every man thine ear, but few thy voice.", source: "Hamlet, I.iii" },
  { line: "The play&rsquo;s the thing.", source: "Hamlet, II.ii" },
  { line: "Brevity is the soul of wit.", source: "Hamlet, II.ii" },
  { line: "Let be.", source: "Hamlet, V.ii" },
  { line: "Our doubts are traitors.", source: "Measure for Measure, I.iv" },
  { line: "For truth is truth to th&rsquo; end of reckoning.", source: "Measure for Measure, V.i" },
  { line: "Wisely and slow; they stumble that run fast.", source: "Romeo and Juliet, II.iii" },
  { line: "All things are ready, if our minds be so.", source: "Henry V, IV.iii" },
  { line: "Men of few words are the best men.", source: "Henry V, III.ii" },
  { line: "How far that little candle throws his beams!", source: "The Merchant of Venice, V.i" },
  { line: "I am not bound to please thee with my answer.", source: "The Merchant of Venice, IV.i" },
  { line: "What&rsquo;s past is prologue.", source: "The Tempest, II.i" },
  { line: "We are such stuff as dreams are made on.", source: "The Tempest, IV.i" },
  { line: "Journeys end in lovers meeting.", source: "Twelfth Night, II.iii" },
  { line: "Come what come may, time and the hour runs through the roughest day.", source: "Macbeth, I.iii" },
  { line: "Nothing will come of nothing.", source: "King Lear, I.i" },
  { line: "Ripeness is all.", source: "King Lear, V.ii" },
  {
    line: "Modest doubt is call&rsquo;d the beacon of the wise.",
    source: "Troilus and Cressida, II.ii",
  },
  { line: "Every why hath a wherefore.", source: "The Comedy of Errors, II.ii" },
  {
    line: "Take but degree away, untune that string, and hark what discord "
      + "follows.",
    source: "Troilus and Cressida, I.iii",
  },
  {
    line: "Oft expectation fails, and most oft there where most it promises.",
    source: "All&rsquo;s Well That Ends Well, II.i",
  },
  { line: "Our remedies oft in ourselves do lie.", source: "All&rsquo;s Well That Ends Well, I.i" },
  { line: "Small have continual plodders ever won.", source: "Love&rsquo;s Labour&rsquo;s Lost, I.i" },
  {
    line: "Study is like the heaven&rsquo;s glorious sun.",
    source: "Love&rsquo;s Labour&rsquo;s Lost, I.i",
  },
  {
    line: "Light seeking light doth light of light beguile.",
    source: "Love&rsquo;s Labour&rsquo;s Lost, I.i",
  },
  { line: "There is nothing either good or bad, but thinking makes it so.", source: "Hamlet, II.ii" },
  { line: "This above all: to thine own self be true.", source: "Hamlet, I.iii" },
  { line: "Suit the action to the word, the word to the action.", source: "Hamlet, III.ii" },
  { line: "To hold as &rsquo;twere the mirror up to nature.", source: "Hamlet, III.ii" },
  { line: "What a piece of work is man!", source: "Hamlet, II.ii" },
  { line: "Conscience does make cowards of us all.", source: "Hamlet, III.i" },
  {
    line: "There&rsquo;s a divinity that shapes our ends, rough-hew them how "
      + "we will.",
    source: "Hamlet, V.ii",
  },
  { line: "More matter, with less art.", source: "Hamlet, II.ii" },
  { line: "I am but mad north-north-west.", source: "Hamlet, II.ii" },
  { line: "Purpose is but the slave to memory.", source: "Hamlet, III.ii" },
  { line: "Our wills and fates do so contrary run.", source: "Hamlet, III.ii" },
  { line: "Use every man after his desert, and who should scape whipping.", source: "Hamlet, II.ii" },
  { line: "Stand and unfold yourself.", source: "Hamlet, I.i" },
  { line: "Not a mouse stirring.", source: "Hamlet, I.i" },
  { line: "Let me not burst in ignorance!", source: "Hamlet, I.iv" },
  {
    line: "And as imagination bodies forth the forms of things unknown, the "
      + "poet&rsquo;s pen turns them to shapes.",
    source: "A Midsummer Night&rsquo;s Dream, V.i",
  },
  {
    line: "The lunatic, the lover, and the poet are of imagination all "
      + "compact.",
    source: "A Midsummer Night&rsquo;s Dream, V.i",
  },
  { line: "The best in this kind are but shadows.", source: "A Midsummer Night&rsquo;s Dream, V.i" },
  { line: "I have had a most rare vision.", source: "A Midsummer Night&rsquo;s Dream, IV.i" },
  {
    line: "The fool doth think he is wise, but the wise man knows himself to "
      + "be a fool.",
    source: "As You Like It, V.i",
  },
  {
    line: "Finds tongues in trees, books in the running brooks, sermons in "
      + "stones, and good in everything.",
    source: "As You Like It, II.i",
  },
  { line: "Sweet are the uses of adversity.", source: "As You Like It, II.i" },
  { line: "Can one desire too much of a good thing?", source: "As You Like It, IV.i" },
  { line: "All the world&rsquo;s a stage.", source: "As You Like It, II.vii" },
  { line: "Men should be what they seem.", source: "Othello, III.iii" },
  {
    line: "Trifles light as air are to the jealous confirmations strong as "
      + "proofs of holy writ.",
    source: "Othello, III.iii",
  },
  { line: "&rsquo;Tis in ourselves that we are thus or thus.", source: "Othello, I.iii" },
  { line: "Our bodies are gardens, to the which our wills are gardeners.", source: "Othello, I.iii" },
  {
    line: "The fault, dear Brutus, is not in our stars, but in ourselves.",
    source: "Julius Caesar, I.ii",
  },
  { line: "There is a tide in the affairs of men.", source: "Julius Caesar, IV.iii" },
  { line: "Men at some time are masters of their fates.", source: "Julius Caesar, I.ii" },
  { line: "He thinks too much: such men are dangerous.", source: "Julius Caesar, I.ii" },
  { line: "It is the bright day that brings forth the adder.", source: "Julius Caesar, II.i" },
  { line: "All the interim is like a phantasma or a hideous dream.", source: "Julius Caesar, II.i" },
  {
    line: "Things done well and with a care, exempt themselves from fear.",
    source: "Henry VIII, I.ii",
  },
  { line: "All that glisters is not gold.", source: "The Merchant of Venice, II.vii" },
  { line: "The devil can cite Scripture for his purpose.", source: "The Merchant of Venice, I.iii" },
  { line: "It is a wise father that knows his own child.", source: "The Merchant of Venice, II.ii" },
  { line: "Truth will come to light.", source: "The Merchant of Venice, II.ii" },
  { line: "The quality of mercy is not strain&rsquo;d.", source: "The Merchant of Venice, IV.i" },
  { line: "Comparisons are odorous.", source: "Much Ado About Nothing, III.v" },
  { line: "Everyone can master a grief but he that has it.", source: "Much Ado About Nothing, III.ii" },
  {
    line: "The course of true love never did run smooth.",
    source: "A Midsummer Night&rsquo;s Dream, I.i",
  },
  { line: "Lord, what fools these mortals be!", source: "A Midsummer Night&rsquo;s Dream, III.ii" },
  {
    line: "Though she be but little, she is fierce.",
    source: "A Midsummer Night&rsquo;s Dream, III.ii",
  },
  { line: "The empty vessel makes the greatest sound.", source: "Henry V, IV.iv" },
  { line: "Self-love, my liege, is not so vile a sin as self-neglecting.", source: "Henry V, II.iv" },
  { line: "The better part of valour is discretion.", source: "Henry IV Part 1, V.iv" },
  { line: "O, while you live, tell truth and shame the devil!", source: "Henry IV Part 1, III.i" },
  { line: "He will give the devil his due.", source: "Henry IV Part 1, I.ii" },
  { line: "Truth hath a quiet breast.", source: "Richard II, I.iii" },
  { line: "The ripest fruit first falls.", source: "Richard II, II.i" },
  { line: "True hope is swift, and flies with swallow&rsquo;s wings.", source: "Richard III, V.ii" },
  { line: "Nothing is but what is not.", source: "Macbeth, I.iii" },
  { line: "Present fears are less than horrible imaginings.", source: "Macbeth, I.iii" },
  { line: "The instruments of darkness tell us truths.", source: "Macbeth, I.iii" },
  { line: "What&rsquo;s done is done.", source: "Macbeth, III.ii" },
  { line: "Things without all remedy should be without regard.", source: "Macbeth, III.ii" },
  { line: "Fair is foul, and foul is fair.", source: "Macbeth, I.i" },
  { line: "When shall we three meet again?", source: "Macbeth, I.i" },
  {
    line: "When the hurlyburly&rsquo;s done, when the battle&rsquo;s lost and "
      + "won.",
    source: "Macbeth, I.i",
  },
  { line: "Have more than thou showest, speak less than thou knowest.", source: "King Lear, I.iv" },
  { line: "O, reason not the need!", source: "King Lear, II.iv" },
  { line: "Mend your speech a little, lest you may mar your fortunes.", source: "King Lear, I.i" },
  { line: "The wheel is come full circle.", source: "King Lear, V.iii" },
  { line: "The worst is not so long as we can say This is the worst.", source: "King Lear, IV.i" },
  {
    line: "Men must endure their going hence, even as their coming hither.",
    source: "King Lear, V.ii",
  },
  { line: "Hell is empty and all the devils are here!", source: "The Tempest, I.ii" },
  { line: "O brave new world, that has such people in&rsquo;t!", source: "The Tempest, V.i" },
  { line: "The rarer action is in virtue than in vengeance.", source: "The Tempest, V.i" },
  { line: "Be not afeard; the isle is full of noises.", source: "The Tempest, III.ii" },
  { line: "Be not afraid of greatness.", source: "Twelfth Night, II.v" },
  { line: "Better a witty fool than a foolish wit.", source: "Twelfth Night, I.v" },
  { line: "Nothing that is so, is so.", source: "Twelfth Night, IV.i" },
  { line: "Love sought is good, but given unsought is better.", source: "Twelfth Night, III.i" },
  {
    line: "The web of our life is of a mingled yarn, good and ill together.",
    source: "All&rsquo;s Well That Ends Well, IV.iii",
  },
  { line: "Love all, trust a few, do wrong to none.", source: "All&rsquo;s Well That Ends Well, I.i" },
  { line: "It is requir&rsquo;d you do awake your faith.", source: "The Winter&rsquo;s Tale, V.iii" },
  {
    line: "What&rsquo;s gone and what&rsquo;s past help should be past grief.",
    source: "The Winter&rsquo;s Tale, III.ii",
  },
  { line: "Home-keeping youth have ever homely wits.", source: "The Two Gentlemen of Verona, I.i" },
  { line: "Hope is a lover&rsquo;s staff.", source: "The Two Gentlemen of Verona, III.i" },
  { line: "There is something in the wind.", source: "The Comedy of Errors, III.i" },
  { line: "The world&rsquo;s mine oyster.", source: "The Merry Wives of Windsor, II.ii" },
  { line: "No profit grows where is no pleasure ta&rsquo;en.", source: "The Taming of the Shrew, I.i" },
  { line: "Strive mightily, but eat and drink as friends.", source: "The Taming of the Shrew, I.ii" },
  { line: "To gild refined gold, to paint the lily.", source: "King John, IV.ii" },
  { line: "Delays have dangerous ends.", source: "Henry VI Part 1, III.ii" },
  { line: "Suspicion always haunts the guilty mind.", source: "Henry VI Part 3, V.vi" },
  { line: "The tempter or the tempted, who sins most?", source: "Measure for Measure, II.ii" },
  { line: "What&rsquo;s in a name?", source: "Romeo and Juliet, II.ii" },
  { line: "He jests at scars that never felt a wound.", source: "Romeo and Juliet, II.ii" },
  {
    line: "There&rsquo;s beggary in the love that can be reckoned.",
    source: "Antony and Cleopatra, I.i",
  },
  { line: "The nature of bad news infects the teller.", source: "Antony and Cleopatra, I.ii" },
  { line: "My salad days, when I was green in judgment.", source: "Antony and Cleopatra, I.v" },
  { line: "Nature teaches beasts to know their friends.", source: "Coriolanus, II.i" },
  { line: "What is the city but the people?", source: "Coriolanus, III.i" },
  { line: "Some griefs are med&rsquo;cinable.", source: "Cymbeline, III.ii" },
  { line: "There&rsquo;s small choice in rotten apples.", source: "The Taming of the Shrew, I.i" },
  { line: "Small things make base men proud.", source: "Henry VI Part 2, IV.i" },
  { line: "Screw your courage to the sticking-place.", source: "Macbeth, I.vii" },
  { line: "Nothing in his life became him like the leaving it.", source: "Macbeth, I.iv" },
  { line: "I have no spur to prick the sides of my intent.", source: "Macbeth, I.vii" },
  { line: "Sleep that knits up the ravell&rsquo;d sleave of care.", source: "Macbeth, II.ii" },
  { line: "I will wear my heart upon my sleeve.", source: "Othello, I.i" },
  { line: "Who steals my purse steals trash.", source: "Othello, III.iii" },
  {
    line: "A jest&rsquo;s prosperity lies in the ear of him that hears it.",
    source: "Love&rsquo;s Labour&rsquo;s Lost, V.ii",
  },
  {
    line: "I like this place and willingly could waste my time in it.",
    source: "As You Like It, II.iv",
  },
  { line: "Few love to hear the sins they love to act.", source: "Pericles, I.i" },

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
  {
    line: "But will they come when you do call for them?",
    source: "Henry IV Part 1, III.i",
    after: 180,
  },
  {
    line: "Better three hours too soon than a minute too late.",
    source: "The Merry Wives of Windsor, II.ii",
    after: 180,
  },
  { line: "Too swift arrives as tardy as too slow.", source: "Romeo and Juliet, II.vi", after: 180 },
  { line: "These violent delights have violent ends.", source: "Romeo and Juliet, II.vi", after: 180 },
  {
    line: "If it were done when &rsquo;tis done, then &rsquo;twere well it "
      + "were done quickly.",
    source: "Macbeth, I.vii",
    after: 180,
  },
  {
    line: "Haste still pays haste, and leisure answers leisure.",
    source: "Measure for Measure, V.i",
    after: 180,
  },
  {
    line: "How sour sweet music is, when time is broke and no proportion "
      + "kept!",
    source: "Richard II, V.v",
    after: 180,
  },
  {
    line: "If all the year were playing holidays, to sport would be as "
      + "tedious as to work.",
    source: "Henry IV Part 1, I.ii",
    after: 180,
  },
  {
    line: "The miserable have no other medicine but only hope.",
    source: "Measure for Measure, III.i",
    after: 180,
  },
  {
    line: "To mourn a mischief that is past and gone is the next way to draw "
      + "new mischief on.",
    source: "Othello, I.iii",
    after: 180,
  },
  { line: "Patch grief with proverbs.", source: "Much Ado About Nothing, V.i", after: 180 },
  {
    line: "There was never yet philosopher that could endure the toothache "
      + "patiently.",
    source: "Much Ado About Nothing, V.i",
    after: 180,
  },
  { line: "Things past redress are now with me past care.", source: "Richard II, II.iii", after: 180 },
  {
    line: "I am a feather for each wind that blows.",
    source: "The Winter&rsquo;s Tale, II.iii",
    after: 180,
  },
  { line: "The game is up.", source: "Cymbeline, III.iii", after: 180 },
  { line: "I am fortune&rsquo;s fool!", source: "Romeo and Juliet, III.i", after: 180 },
  {
    line: "It is a tale told by an idiot, full of sound and fury, signifying "
      + "nothing.",
    source: "Macbeth, V.v",
    after: 180,
  },
  { line: "Life&rsquo;s but a walking shadow, a poor player.", source: "Macbeth, V.v", after: 180 },
  {
    line: "Stand not upon the order of your going, but go at once.",
    source: "Macbeth, III.iv",
    after: 180,
  },
  { line: "Is this a dagger which I see before me?", source: "Macbeth, II.i", after: 180 },
  {
    line: "Do you see yonder cloud that&rsquo;s almost in shape of a camel?",
    source: "Hamlet, III.ii",
    after: 180,
  },
  { line: "That it should come to this!", source: "Hamlet, I.ii", after: 180 },
  {
    line: "How weary, stale, flat, and unprofitable seem to me all the uses "
      + "of this world!",
    source: "Hamlet, I.ii",
    after: 180,
  },
  {
    line: "I have of late, but wherefore I know not, lost all my mirth.",
    source: "Hamlet, II.ii",
    after: 180,
  },
  { line: "Denmark&rsquo;s a prison.", source: "Hamlet, II.ii", after: 180 },
  { line: "Uneasy lies the head that wears a crown.", source: "Henry IV Part 2, III.i", after: 180 },
  {
    line: "But thoughts, the slaves of life, and life, time&rsquo;s fool.",
    source: "Henry IV Part 1, V.iv",
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
  {
    line: "Time hath, my lord, a wallet at his back, wherein he puts alms for "
      + "oblivion.",
    source: "Troilus and Cressida, III.iii",
    after: 900,
  },
  {
    line: "The end crowns all, and that old common arbitrator, Time, will one "
      + "day end it.",
    source: "Troilus and Cressida, IV.v",
    after: 900,
  },
  { line: "Time&rsquo;s the king of men.", source: "Pericles, II.iii", after: 900 },
  {
    line: "In time we hate that which we often fear.",
    source: "Antony and Cleopatra, I.iii",
    after: 900,
  },
  { line: "A sad tale&rsquo;s best for winter.", source: "The Winter&rsquo;s Tale, II.i", after: 900 },
  { line: "Life is as tedious as a twice-told tale.", source: "King John, III.iv", after: 900 },
  { line: "Fear no more the heat o&rsquo; th&rsquo; sun.", source: "Cymbeline, IV.ii", after: 900 },
  {
    line: "The gaudy, blabbing, and remorseful day is crept into the bosom of "
      + "the sea.",
    source: "Henry VI Part 2, IV.i",
    after: 900,
  },
  { line: "Out, out, brief candle!", source: "Macbeth, V.v", after: 900 },
  { line: "I have almost forgot the taste of fears.", source: "Macbeth, V.v", after: 900 },
  { line: "True is it that we have seen better days.", source: "As You Like It, II.vii", after: 900 },
  { line: "Our revels now are ended.", source: "The Tempest, IV.i", after: 900 },
  {
    line: "Let us not burden our remembrances with a heaviness that&rsquo;s "
      + "gone.",
    source: "The Tempest, V.i",
    after: 900,
  },
  { line: "Full fathom five thy father lies.", source: "The Tempest, I.ii", after: 900 },
  { line: "Alas, poor Yorick!", source: "Hamlet, V.i", after: 900 },
  { line: "The cat will mew, and dog will have his day.", source: "Hamlet, V.i", after: 900 },
  { line: "A horse! a horse! my kingdom for a horse!", source: "Richard III, V.iv", after: 900 },

  // An hour. Whatever this is, it is not being watched.
  { line: "The rest is silence.", source: "Hamlet, V.ii", after: 3600 },
  { line: "If it be now, &rsquo;tis not to come.", source: "Hamlet, V.ii", after: 3600 },
  { line: "Now my charms are all o&rsquo;erthrown.", source: "The Tempest, Epilogue", after: 3600 },
  { line: "Men&rsquo;s evil manners live in brass.", source: "Henry VIII, IV.ii", after: 3600 },
  {
    line: "What&rsquo;s yet in this that bears the name of life?",
    source: "Measure for Measure, III.i",
    after: 3600,
  },
  {
    line: "And thus the whirligig of time brings in his revenges.",
    source: "Twelfth Night, V.i",
    after: 3600,
  },
  {
    line: "All&rsquo;s well that ends well.",
    source: "All&rsquo;s Well That Ends Well, IV.iv",
    after: 3600,
  },
  {
    line: "Then come kiss me, sweet and twenty; youth&rsquo;s a stuff will "
      + "not endure.",
    source: "Twelfth Night, II.iii",
    after: 3600,
  },
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
