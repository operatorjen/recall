import readline from 'node:readline'
import { stdin as input, stdout as output } from 'node:process'
import { createRelationalModule, RelationalUtils } from 'relational'
import { Funnels } from 'funnels'
import { Merger } from 'merger'
import { createAcquisitionModule } from 'acquisition'
import { createRecallModule } from './index.mjs'

function banner(title) {
  const line = '='.repeat(title.length, 4)
  console.log(`\n${line}\n  ${title}\n${line}\n`)
}

function pr(obj) { return JSON.stringify(obj, null, 2) }

function createOfflineMerger(relational) {
  const merger = new Merger({}, relational, { defaultModel: 'demo-offline' })
  if (!Array.isArray(merger.configs)) merger.configs = []
  return merger
}

if (typeof Funnels.prototype.applyBias !== 'function') {
  Funnels.prototype.applyBias = function applyBias(agentId, bias) {
    if (!bias) return
    if (!this.agentBiases) this.agentBiases = new Map()
    this.agentBiases.set(agentId, bias)
    const eS = typeof bias.energyScale === 'number' ? bias.energyScale : 1
    const nS = typeof bias.noveltyScale === 'number' ? bias.noveltyScale : 1
    const lr = 0.05
    this.gs.energyMean = this.gs.energyMean * (1 - lr) + (this.gs.energyMean * eS) * lr
    this.gs.energyVariance = this.gs.energyVariance * (1 - lr) + (this.gs.energyVariance * nS) * lr
  }
}

async function main() {
  banner('CYBERNETIC MEMETIC CONVERSATIONS – INTERACTIVE DEMO')
  const userId = 'Me'
  const agentId = 'Agent'
  const agents = [
    RelationalUtils.createAgentConfig(userId, {
      name: 'You (external)',
      plasticity: 0.7,
      energy: 0.6,
      socialStyle: 'observer'
    }),
    RelationalUtils.createAgentConfig(agentId, {
      name: 'Alpha',
      plasticity: 0.8,
      energy: 0.75,
      socialStyle: 'balanced',
      themePacks: {},
      motifConfig: {},
      initState: {}
    })
  ]

  const relational = createRelationalModule(agents, { tasteConfig: { ProjectionsCtor: undefined } })
  const funnels = new Funnels()
  const merger = createOfflineMerger(relational)
  const { acquisition } = createAcquisitionModule(merger, relational, {
    acceptThreshold: 0.05,
    deferThreshold: 0.05
  })
  const { recall } = createRecallModule(relational, merger, funnels, {
    minSamplesForBias: 2,
    maxEnergyScale: 0.6,
    maxNoveltyBias: 0.5,
    learningAggressiveness: 1.0
  })
  const lastMergeBySpeaker = Object.create(null)
  console.log('Interactive wiring complete.\n')
  console.log('You are talking to agent:', agentId)
  console.log('Type messages and press enter. Commands:\n')
  console.log('  /bias      – show current Recall bias for the agent')
  console.log('  /funnels   – show current Funnels global state')
  console.log('  /rel       – show relational snapshot (you <-> agent)')
  console.log('  /stats     – show acquisition stats')
  console.log('  /lex       – show latest stance/template/lexicon for you   agent')
  console.log('  /exit      – quit demo\n')
  const rl = readline.createInterface({ input, output })
  const ask = (q) => new Promise((resolve) => rl.question(q, resolve))

  async function acquireAndRemember({ text, speakerId, targetId, sourceType, direction }) {
    const result = await acquisition.consider({
      text,
      speakerId,
      targetId,
      direction,
      sourceType,
      channels: ['cli']
    })
    if (result.decision === 'accept' && result.mergerResult) {
      const { stance, template, lexicon } = result.mergerResult
      lastMergeBySpeaker[speakerId] = { stance, template, lexicon }
      recall.recordAcquisition({
        speakerId,
        targetId,
        stance,
        template,
        lexicon,
        score: result.score,
        sourceType,
        channels: ['cli'],
        snapshot: result.snapshot
      })
      const bias = recall.applyFunnelBias(speakerId)
      return { result, bias }
    }
    return { result, bias: null }
  }

  async function handleCommand(cmd) {
    switch (cmd) {
      case '/bias': {
        const profile = recall.getAgentProfile(agentId)
        const bias = recall.getFunnelBias(agentId)
        console.log('\n[Recall bias for agent]', agentId)
        console.log(pr({ bias, samples: profile?.samples ?? 0 }))
        break
      }
      case '/funnels': {
        console.log('\n[Funnels global state]')
        console.log(pr(funnels.getGlobalState()))
        break
      }
      case '/rel': {
        try {
          const rel = relational.getInteraction(userId, agentId)
          console.log('\n[Relational snapshot you -> agent]')
          console.log(pr(rel.state))
        } catch {
          console.log('\n[Relational snapshot not available yet]')
        }
        break
      }
      case '/stats': {
        console.log('\n[Acquisition stats]')
        console.log(pr(acquisition.getStats()))
        break
      }
      case '/lex': {
        const aL = lastMergeBySpeaker[agentId] || null
        const uL = lastMergeBySpeaker[userId] || null
        const nB = merger.getDebugBucket('neutral')
        const sB = merger.getDebugBucket('supportive')
        const dB = merger.getDebugBucket('defensive')
        console.log('\n[Most recent merged stance/template/lexicon]')
        console.log(
          pr({
            agent: aL || '(none yet)',
            user: uL || '(none yet)'
          })
        )
        console.log('\n[Merger buckets – learned templates   lexicon samples]')
        console.log(pr({ neutral: nB || '(empty)', supportive: sB || '(empty)', defensive: dB || '(empty)' }))
        break
      }
      default:
        console.log('Unknown command.')
    }
  }

  async function handleTurn(userText) {
    await acquireAndRemember({
      text: userText,
      speakerId: userId,
      targetId: agentId,
      sourceType: 'user',
      direction: 'incoming'
    })
    const genCfg = recall.getGenerationOverrides(agentId, userId, { stance: 'neutral' })
    const tR = await relational.processTurn(agentId, userText, {
      fromUserId: userId,
      generationConfig: genCfg
    })
    const aT = tR.baseResponse?.text || '(no text generated)'
    console.log(`\n${agentId}: ${aT}`)
    const agentAcq = await acquireAndRemember({
      text: aT,
      speakerId: agentId,
      targetId: userId,
      sourceType: 'internal',
      direction: 'outgoing'
    })
    const tC = Math.floor(Math.random() * 6) + 1
    funnels.update(tC)
    const bias = recall.getFunnelBias(agentId)
    console.log(`\n[Turn telemetry x ${tC}]`)
    console.log(pr({
      agentAcquisition: {
        decision: agentAcq.result.decision,
        score: Number(agentAcq.result.score.toFixed(3))
      },
      recallBias: {
        energyScale: Number(bias.energyScale.toFixed(3)),
        noveltyScale: Number(bias.noveltyScale.toFixed(3)),
        expressivenessScale: Number(bias.expressivenessScale.toFixed(3)),
        statePreference: bias.statePreference
      },
      funnelsEnergyMean: funnels.getGlobalState().energyMean,
      generationConfig: {
        source: genCfg.source,
        stance: genCfg.stance,
        templateSample: Array.isArray(genCfg.templates) ? genCfg.templates.slice(0, 3) : null
      }
    }))
    console.log('')
  }

  for (;;) {
    const line = await ask(`\n${'*'.repeat(50)}\nMe: `)
    if (!line) continue
    const trimmed = line.trim()
    if (trimmed === '/exit') break
    if (trimmed.startsWith('/')) {
      await handleCommand(trimmed)
      continue
    }
    await handleTurn(trimmed)
  }
  rl.close()
  console.log('\nFinal diagnostics:')
  console.log('\n[Acquisition stats]')
  console.log(pr(acquisition.getStats()))
  console.log('\n[Recall profile for agent]', agentId)
  console.log(pr(recall.getAgentProfile(agentId) || {}))
  console.log('\n[Funnels global state]')
  console.log(pr(funnels.getGlobalState()))
  console.log('\n[Network social dynamics]')
  console.log(pr(relational.getSocialDynamics()))
  console.log('\nDemo complete.\n')
}

main().catch((err) => {
  console.error('\nDemo error:', err)
  process.exit(1)
})