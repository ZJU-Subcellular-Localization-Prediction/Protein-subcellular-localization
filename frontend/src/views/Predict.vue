<template>
  <div class="predict-page">
    <el-page-header @back="$router.push('/')" title="Back">
      <template #content>
        <span class="page-title">Subcellular Localization Prediction</span>
      </template>
    </el-page-header>

    <!-- Input -->
    <el-card class="section-card">
      <p class="section-hint">
        Enter an amino acid sequence or upload a FASTA file.
      </p>
      <SequenceInput
        @submit="handleSubmit"
        @clear="handleClear"
      />
    </el-card>

    <!-- Loading skeleton -->
    <div v-if="loading" class="loading-section">
      <el-skeleton :rows="4" animated :throttle="500" />
      <div class="loading-text">
        <el-icon class="is-loading" :size="18"><Loading /></el-icon>
        {{ loadingHint }}
      </div>
      <el-progress :percentage="loadingPercent" :indeterminate="loadingPercent < 10" :duration="1" />
    </div>

    <!-- Error -->
    <el-alert
      v-if="error"
      :title="error"
      type="error"
      :closable="true"
      show-icon
      class="error-alert"
      @close="error = null"
    />

    <!-- Results -->
    <template v-if="result && !loading">
      <ResultCard
        :location="result.predicted_location"
        :confidence="result.location_confidence"
        :membrane="result.predicted_membrane"
        :membrane-confidence="result.membrane_confidence"
        :inference-time-ms="result.inference_time_ms"
        :sequence-id="result.sequence_id"
        :model-version="result.model_version"
      />

      <el-row :gutter="20" class="viz-row">
        <el-col :md="12" :sm="24">
          <CellDiagram
            :highlight-location="result.predicted_location"
            :all-probabilities="result.all_probabilities"
          />
        </el-col>
        <el-col :md="12" :sm="24">
          <ProbabilityChart :probabilities="result.all_probabilities" />
        </el-col>
      </el-row>

      <AttentionHeatmap
        v-if="result.attention_weights && result.attention_weights.length > 0"
        :attention-weights="result.attention_weights"
      />

      <div class="history-link">
        <el-button text type="primary" @click="$router.push('/history')">
          <el-icon><Clock /></el-icon>
          View Prediction History
        </el-button>
      </div>
    </template>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { ElMessage } from 'element-plus'
import SequenceInput from '@/components/SequenceInput.vue'
import ResultCard from '@/components/ResultCard.vue'
import ProbabilityChart from '@/components/ProbabilityChart.vue'
import CellDiagram from '@/components/CellDiagram.vue'
import AttentionHeatmap from '@/components/AttentionHeatmap.vue'
import { postPredict } from '@/api/predict'

// ---- state ----
const loading = ref(false)
const error = ref(null)
const result = ref(null)

// ---- loading hints (cycle through phases) ----
const loadingHint = ref('')
const loadingPercent = ref(0)
let hintTimer = null
let progressTimer = null

function startLoadingAnimation() {
  const hints = [
    'Cleaning sequence...',
    'Running ESM-2 feature extraction (640-dim, 150M params)...',
    'Model inference in progress...',
    'Almost done...'
  ]
  let step = 0
  loadingHint.value = hints[0]
  loadingPercent.value = 5

  hintTimer = setInterval(() => {
    step++
    if (step < hints.length) {
      loadingHint.value = hints[step]
    }
  }, 3000)

  progressTimer = setInterval(() => {
    if (loadingPercent.value < 95) {
      loadingPercent.value += Math.random() * 10
      if (loadingPercent.value > 95) loadingPercent.value = 95
    }
  }, 1500)
}

function finishLoadingAnimation() {
  clearInterval(hintTimer)
  clearInterval(progressTimer)
  hintTimer = null
  progressTimer = null
  loadingPercent.value = 100
}

// ---- handlers ----
async function handleSubmit(sequence) {
  error.value = null
  result.value = null
  loading.value = true
  startLoadingAnimation()

  try {
    const resp = await postPredict(sequence)
    result.value = resp
    ElMessage.success(`Prediction complete — ${resp.predicted_location} (${(resp.location_confidence * 100).toFixed(1)}%) in ${resp.inference_time_ms}ms`)
  } catch (e) {
    const msg = e.message || 'Unknown error occurred'
    error.value = msg
    ElMessage.error(msg)
  } finally {
    finishLoadingAnimation()
    // brief delay so user sees progress hit 100% before result renders
    setTimeout(() => { loading.value = false }, 350)
  }
}

function handleClear() {
  error.value = null
  result.value = null
}
</script>

<style scoped>
.predict-page {
  max-width: 900px;
  margin: 0 auto;
}

.page-title {
  font-size: 18px;
  font-weight: 600;
}

.section-card {
  margin-top: 24px;
}

.section-hint {
  font-size: 14px;
  color: #909399;
  margin-bottom: 16px;
}

.loading-section {
  margin-top: 24px;
  padding: 24px;
  background: #fff;
  border-radius: 8px;
  border: 1px solid #e4e7ed;
}

.loading-text {
  display: flex;
  align-items: center;
  gap: 8px;
  margin: 16px 0 12px;
  font-size: 14px;
  color: #606266;
}

.error-alert {
  margin-top: 24px;
}

.viz-row {
  margin-top: 20px;
}

.history-link {
  text-align: center;
  margin: 32px 0 16px;
}
</style>
