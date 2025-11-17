const DEFAULT_MIN_SAMPLES_FOR_BIAS = 5
const DEFAULT_STANCE_WEIGHT_DEFENSIVE = -1
const DEFAULT_STANCE_WEIGHT_NEUTRAL = 0
const DEFAULT_STANCE_WEIGHT_SUPPORTIVE = 1
const DEFAULT_DECAY_PER_MS = 0
const DEFAULT_MAX_STATE_BIAS = 0.4
const DEFAULT_MAX_ENERGY_SCALE = 0.4
const DEFAULT_MAX_NOVELTY_BIAS = 0.3
const DEFAULT_LEARNING_AGGRESSIVENESS = 1.0
const RELATIONAL_DEFAULT_LEVEL = 0.5
const TEMPLATE_RICHNESS_MAX_TEMPLATES = 20
const NET_TONE_MIN = -1
const NET_TONE_MAX = 1
const ENERGY_NORMALIZATION_FACTOR = 2
const ENERGY_NORMALIZATION_SHIFT = 1
const ENERGY_CENTER = 0.5
const STATE_PREF_TONE_WEIGHT = 0.7
const STATE_PREF_ENERGY_WEIGHT = 0.3
const DECAY_FACTOR_THRESHOLD = 0.999
const LEXICON_DEFAULT_WEIGHT = 1
const NEUTRAL_STATE_WEIGHT = 1 / 3
const EXPRESSIVENESS_BLEND = 0.5

export class Recall {
  constructor(relational = null, merger = null, funnels = null, options = {}) {
    this.relational = relational
    this.merger = merger
    this.funnels = funnels
    this.cfg = {
      minSamplesForBias: options.minSamplesForBias ?? DEFAULT_MIN_SAMPLES_FOR_BIAS,
      stanceWeights: {
        defensive: DEFAULT_STANCE_WEIGHT_DEFENSIVE,
        neutral: DEFAULT_STANCE_WEIGHT_NEUTRAL,
        supportive: DEFAULT_STANCE_WEIGHT_SUPPORTIVE,
        ...(options.stanceWeights || {})
      },
      decayPerMs: options.decayPerMs ?? DEFAULT_DECAY_PER_MS,
      maxStateBias: options.maxStateBias ?? DEFAULT_MAX_STATE_BIAS,
      maxEnergyScale: options.maxEnergyScale ?? DEFAULT_MAX_ENERGY_SCALE,
      maxNoveltyBias: options.maxNoveltyBias ?? DEFAULT_MAX_NOVELTY_BIAS,
      learningAggressiveness: options.learningAggressiveness ?? DEFAULT_LEARNING_AGGRESSIVENESS
    }
    this.agentProfiles = new Map()
  }

  _computesB(profile) {
    const counts = profile.stanceCounts
    const total = (counts.defensive || 0) + (counts.neutral || 0) + (counts.supportive || 0)
    if (!total) return { nT: 0, dF: 0, sF: 0 }
    const dF = (counts.defensive || 0) / total
    const sF = (counts.supportive || 0) / total
    const w = this.cfg.stanceWeights
    const weightedSum = dF * (w.defensive ?? DEFAULT_STANCE_WEIGHT_DEFENSIVE) + ((counts.neutral || 0) / total) * (w.neutral ?? DEFAULT_STANCE_WEIGHT_NEUTRAL) + sF * (w.supportive ?? DEFAULT_STANCE_WEIGHT_SUPPORTIVE)
    const nT = this._clamp(weightedSum, NET_TONE_MIN, NET_TONE_MAX)
    return { nT, dF, sF }
  }

  _computerB(profile) {
    const r = profile.relational
    const trust = r.trustSamples > 0 ? r.trustSum / r.trustSamples : RELATIONAL_DEFAULT_LEVEL
    const comfort = r.comfortSamples > 0 ? r.comfortSum / r.comfortSamples : RELATIONAL_DEFAULT_LEVEL
    const alignment = r.alignmentSamples > 0 ? r.alignmentSum / r.alignmentSamples : RELATIONAL_DEFAULT_LEVEL
    const energy = r.energySamples > 0 ? r.energySum / r.energySamples : RELATIONAL_DEFAULT_LEVEL
    return {
      trust: this._clamp(trust, 0, 1),
      comfort: this._clamp(comfort, 0, 1),
      alignment: this._clamp(alignment, 0, 1),
      energy: this._clamp(energy, 0, 1)
    }
  }

  _computesBi(profile) {
    const tC = Object.keys(profile.templates).length
    let uT = 0, totalCounts = 0
    const lex = profile.lexiconCounts
    for (const pos of Object.keys(lex)) {
      const bucket = lex[pos]
      uT += Object.keys(bucket).length
      for (const w of Object.keys(bucket)) { totalCounts += bucket[w] }
    }
    const ri = uT + tC
    const novelty = totalCounts > 0 ? ri / (totalCounts + ri) : 0
    const tR = tC > 0 ? Math.min(1, tC / TEMPLATE_RICHNESS_MAX_TEMPLATES) : 0
    return { novelty, tR }
  }

  _mapToneToStatePreference(nT, eL) {
    const tone = this._clamp(nT, NET_TONE_MIN, NET_TONE_MAX)
    const energy = this._clamp(eL * ENERGY_NORMALIZATION_FACTOR - ENERGY_NORMALIZATION_SHIFT, NET_TONE_MIN, NET_TONE_MAX)
    const rut = this._clamp(-tone * STATE_PREF_TONE_WEIGHT + (1 - (energy + ENERGY_NORMALIZATION_SHIFT) / ENERGY_NORMALIZATION_FACTOR) * STATE_PREF_ENERGY_WEIGHT, 0, 1)
    const growing = this._clamp(tone * STATE_PREF_TONE_WEIGHT + ((energy + ENERGY_NORMALIZATION_SHIFT) / ENERGY_NORMALIZATION_FACTOR) * STATE_PREF_ENERGY_WEIGHT, 0, 1)
    let emerging = 1 - (rut + growing)
    if (emerging < 0) emerging = 0
    const sum = rut + emerging + growing || 1
    return {
      rut: rut / sum,
      emerging: emerging / sum,
      growing: growing / sum
    }
  }

  _clamp(x, min, max) {
    if (x < min) return min
    if (x > max) return max
    return x
  }

  _ensureProfile(agentId) {
    let profile = this.agentProfiles.get(agentId)
    if (!profile) {
      profile = {
        samples: 0,
        lastUpdated: Date.now(),
        stanceCounts: { defensive: 0, neutral: 0, supportive: 0 },
        templates: {},
        lexiconCounts: {
          nouns: {},
          verbs: {},
          adjectives: {},
          adverbs: {},
          conjunctions: {},
          pronouns: {},
          articles: {},
          prepositions: {},
          auxiliaries: {},
          modals: {}
        },
        scoreSum: 0,
        scoreSqSum: 0,
        sourceCounts: {},
        channelCounts: {},
        targets: {},
        relational: {
          trustSum: 0,
          trustSamples: 0,
          comfortSum: 0,
          comfortSamples: 0,
          alignmentSum: 0,
          alignmentSamples: 0,
          energySum: 0,
          energySamples: 0,
          stanceBandCounts: {
            defensive: 0,
            neutral: 0,
            supportive: 0
          }
        }
      }
      this.agentProfiles.set(agentId, profile)
    }
    return profile
  }

  _maybeDecayProfile(profile, now) {
    if (!this.cfg.decayPerMs || !profile) return
    const dt = now - profile.lastUpdated
    if (dt <= 0) return
    const factor = Math.exp(-this.cfg.decayPerMs * dt)
    if (factor >= DECAY_FACTOR_THRESHOLD) return
    profile.samples *= factor
    profile.scoreSum *= factor
    profile.scoreSqSum *= factor
    const decHistogram = (obj) => { for (const k of Object.keys(obj)) { obj[k] *= factor } }
    decHistogram(profile.stanceCounts)
    decHistogram(profile.sourceCounts)
    decHistogram(profile.channelCounts)
    decHistogram(profile.targets)
    const decLexicon = (lex) => {
      for (const pos of Object.keys(lex)) {
        const bucket = lex[pos]
        for (const w of Object.keys(bucket)) { bucket[w] *= factor }
      }
    }
    decLexicon(profile.lexiconCounts)
    const r = profile.relational
    r.trustSum *= factor
    r.trustSamples *= factor
    r.comfortSum *= factor
    r.comfortSamples *= factor
    r.alignmentSum *= factor
    r.alignmentSamples *= factor
    r.energySum *= factor
    r.energySamples *= factor
    decHistogram(r.stanceBandCounts)
  }

  _accumulateLexicon(lexCounts, lexicon, weight) {
    const w = typeof weight === 'number' ? weight : LEXICON_DEFAULT_WEIGHT
    const posList = ['nouns', 'verbs', 'adjectives', 'adverbs', 'conjunctions', 'pronouns', 'articles', 'prepositions', 'auxiliaries', 'modals']
    for (const pos of posList) {
      const src = Array.isArray(lexicon[pos]) ? lexicon[pos] : []
      const dst = lexCounts[pos]
      for (const token of src) {
        if (!token) continue
        dst[token] = (dst[token] || 0) + w
      }
    }
  }

  _mapRelationalStanceToBand(relationalStance) {
    const s = (relationalStance || '').toLowerCase()
    if (s === 'defensive') return 'defensive'
    if (s === 'cautious') return 'neutral'
    if (s === 'collaborative') return 'supportive'
    if (s === 'intimate') return 'supportive'
    return 'neutral'
  }

  _warmupFactor(profile) {
    if (!profile || profile.samples <= 0) return 0
    const base = profile.samples / this.cfg.minSamplesForBias
    const clipped = this._clamp(base, 0, 1)
    return clipped * this.cfg.learningAggressiveness
  }

  getGenerationOverrides(speakerId, targetId = null, fallback = {}) {
    const fS = fallback.stance || 'neutral'
    const fB = fallback.templates || null
    const fL = fallback.lexicon || null
    if (!this.merger || typeof this.merger.getGenerationConfig !== 'function') {
      return {
        stance: fS,
        templates: fB,
        lexicon: fL,
        source: 'fallback:no-merger'
      }
    }
    try {
      const cfg = this.merger.getGenerationConfig(speakerId, targetId, { stance: fS, templates: fB || [], lexicon: fL || {} }) || {}
      const templates = Array.isArray(cfg.templates) ? cfg.templates : fB
      const lexicon = cfg.lexicon && typeof cfg.lexicon === 'object' ? cfg.lexicon : fL
      return {
        stance: cfg.stance || fS,
        templates,
        lexicon,
        source: 'merger'
      }
    } catch {
      return {
        stance: fS,
        templates: fB,
        lexicon: fL,
        source: 'fallback:error'
      }
    }
  }

  recordAcquisition(payload) {
    if (!payload || !payload.speakerId) return
    const {
      speakerId,
      targetId = null,
      stance = 'neutral',
      template = '',
      lexicon = {},
      score = 0,
      sourceType = 'internal',
      channels = [],
      snapshot = null,
      timestamp = Date.now()
    } = payload
    const profile = this._ensureProfile(speakerId)
    this._maybeDecayProfile(profile, timestamp)
    profile.samples += 1
    profile.lastUpdated = timestamp
    profile.stanceCounts[stance] = (profile.stanceCounts[stance] || 0) + 1
    if (targetId) {
      const key = targetId
      profile.targets[key] = (profile.targets[key] || 0) + 1
    }
    profile.scoreSum += score
    profile.scoreSqSum += score * score
    if (template && typeof template === 'string') profile.templates[template] = (profile.templates[template] || 0) + 1
    this._accumulateLexicon(profile.lexiconCounts, lexicon, score)
    profile.sourceCounts[sourceType] = (profile.sourceCounts[sourceType] || 0) + 1
    for (const ch of channels || []) {
      if (!ch) continue
      profile.channelCounts[ch] = (profile.channelCounts[ch] || 0) + 1
    }
    if (snapshot && typeof snapshot === 'object') {
      const { trust, comfort, alignment, energy, stanceBand } = snapshot
      if (typeof trust === 'number') {
        profile.relational.trustSum += trust
        profile.relational.trustSamples += 1
      }
      if (typeof comfort === 'number') {
        profile.relational.comfortSum += comfort
        profile.relational.comfortSamples += 1
      }
      if (typeof alignment === 'number') {
        profile.relational.alignmentSum += alignment
        profile.relational.alignmentSamples += 1
      }
      if (typeof energy === 'number') {
        profile.relational.energySum += energy
        profile.relational.energySamples += 1
      }
      if (stanceBand) profile.relational.stanceBandCounts[stanceBand] = (profile.relational.stanceBandCounts[stanceBand] || 0) + 1
    }
  }

  getAgentProfile(agentId) { return this.agentProfiles.get(agentId) || null }

  getFunnelBias(agentId) {
    const profile = this.agentProfiles.get(agentId)
    if (!profile || profile.samples <= 0) {
      return {
        energyScale: 1,
        noveltyScale: 1,
        expressivenessScale: 1,
        statePreference: { rut: 0, emerging: 0, growing: 0 }
      }
    }
    const sB = this._computesB(profile)
    const rB = this._computerB(profile)
    const sBi = this._computesBi(profile)
    const nT = sB.nT
    const eL = rB.energy
    const rSp = this._mapToneToStatePreference(nT, eL)
    const warmup = this._warmupFactor(profile)
    const maxE = this.cfg.maxEnergyScale
    const maxN = this.cfg.maxNoveltyBias
    const rEd = this._clamp((eL - ENERGY_CENTER) * ENERGY_NORMALIZATION_FACTOR * maxE, -maxE, maxE)
    const rNd = this._clamp(sBi.novelty * maxN, -maxN, maxN)
    const rEx = this._clamp((nT + sBi.tR) * EXPRESSIVENESS_BLEND * maxE, -maxE, maxE)
    const energyScale = 1 + rEd * warmup
    const noveltyScale = 1 + rNd * warmup
    const eS = 1 + rEx * warmup
    const neutralState = {
      rut: NEUTRAL_STATE_WEIGHT,
      emerging: NEUTRAL_STATE_WEIGHT,
      growing: NEUTRAL_STATE_WEIGHT
    }
    const statePreference = {
      rut: neutralState.rut * (1 - warmup) + rSp.rut * warmup,
      emerging: neutralState.emerging * (1 - warmup) + rSp.emerging * warmup,
      growing: neutralState.growing * (1 - warmup) + rSp.growing * warmup
    }
    return {
      energyScale,
      noveltyScale,
      expressivenessScale: eS,
      statePreference
    }
  }

  applyParameters(parameters = {}) {
    if (!parameters || typeof parameters !== 'object') return
    const { recall, funnels } = parameters
    if (recall && typeof recall === 'object') {
      for (const k of Object.keys(recall)) {
        const v = recall[k]
        if (k === 'stanceWeights' && v && typeof v === 'object') {
          this.cfg.stanceWeights = { ...this.cfg.stanceWeights, ...v }
        } else if (k in this.cfg && typeof v === 'number') {
          this.cfg[k] = v
        }
      }
    }
    if (funnels && this.funnels) {
      if (typeof this.funnels.applyParameters === 'function') {
        this.funnels.applyParameters(funnels)
      } else if (typeof this.funnels.applyTheme === 'function') {
        this.funnels.applyTheme(funnels)
      }
    }
  }

  applyLexiconSyntaxOverrides({ lexicon, syntax } = {}) {
    if (!this.merger) return
    if (typeof this.merger.applyLexiconSyntaxOverrides === 'function') {
      this.merger.applyLexiconSyntaxOverrides({ lexicon, syntax })
    } else if (typeof this.merger.applyPersona === 'function') {
      this.merger.applyPersona({ lexicon, syntax })
    }
  }

  applyPersona(persona) {
    if (!persona || typeof persona !== 'object') return
    this.persona = persona
    if (this.merger) {
      if (typeof this.merger.applyPersona === 'function') {
        this.merger.applyPersona(persona)
      } else if (typeof this.merger.applyLexiconSyntaxOverrides === 'function') {
        const { lexicon, syntax } = persona
        this.merger.applyLexiconSyntaxOverrides({ lexicon, syntax })
      }
    }
    if (persona.parameters && typeof this.applyParameters === 'function') this.applyParameters(persona.parameters)
  }

  applyTheme(theme) {
    if (!theme || typeof theme !== 'object') return
    if (theme.recall && typeof theme.recall === 'object') this.applyParameters({ recall: theme.recall })
    if (this.relational && typeof this.relational.applyTheme === 'function') this.relational.applyTheme(theme)
    if (this.funnels && typeof this.funnels.applyTheme === 'function') this.funnels.applyTheme(theme)
  }

  applyFunnelBias(agentId) {
    if (!this.funnels || typeof this.funnels.applyBias !== 'function') return this.getFunnelBias(agentId)
    const bias = this.getFunnelBias(agentId)
    this.funnels.applyBias(agentId, bias)
    return bias
  }
}

export function createRecallModule(relational = null, merger = null, funnels = null, options = {}) {
  const recall = new Recall(relational, merger, funnels, options)
  return { recall }
}