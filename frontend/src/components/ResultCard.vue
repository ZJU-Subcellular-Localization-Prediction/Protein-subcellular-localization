<template>
  <el-card class="result-card" shadow="hover">
    <template #header>
      <div class="card-header">
        <span>Prediction Result</span>
        <el-tag v-if="modelVersion" size="small" type="info">{{ modelVersion }}</el-tag>
      </div>
    </template>

    <el-row :gutter="24">
      <!-- Location result -->
      <el-col :span="12" class="result-col">
        <div class="result-label">Subcellular Location</div>
        <el-tag
          :color="locationColor"
          size="large"
          effect="dark"
          class="location-tag"
        >
          {{ location }}
        </el-tag>
        <div class="confidence-ring">
          <el-progress
            type="dashboard"
            :percentage="confidencePercent"
            :color="progressColor"
            :stroke-width="12"
            :width="120"
          >
            <template #default="{ percentage }">
              <span class="percentage-text">{{ percentage }}%</span>
            </template>
          </el-progress>
        </div>
      </el-col>

      <!-- Membrane result -->
      <el-col :span="12" class="result-col">
        <div class="result-label">Membrane Binding</div>
        <el-tag
          v-if="membrane"
          :type="membraneType"
          size="large"
          effect="plain"
          class="membrane-tag"
        >
          {{ membrane }}
        </el-tag>
        <el-tag v-else type="info" size="large" effect="plain">
          N/A
        </el-tag>

        <div v-if="membraneConfidence != null" class="membrane-confidence">
          <span class="membrane-pct">{{ (membraneConfidence * 100).toFixed(1) }}%</span>
          <span class="membrane-label">confidence</span>
        </div>

        <el-descriptions
          :column="1"
          border
          size="small"
          class="info-table"
        >
          <el-descriptions-item label="Inference time">
            {{ inferenceTimeMs }} ms
          </el-descriptions-item>
          <el-descriptions-item label="Sequence ID">
            <code class="seq-id">{{ sequenceId }}</code>
          </el-descriptions-item>
        </el-descriptions>
      </el-col>
    </el-row>
  </el-card>
</template>

<script setup>
import { computed } from 'vue'

const props = defineProps({
  location: { type: String, required: true },
  confidence: { type: Number, required: true },
  membrane: { type: String, default: null },
  membraneConfidence: { type: Number, default: null },
  inferenceTimeMs: { type: Number, default: 0 },
  sequenceId: { type: String, default: '' },
  modelVersion: { type: String, default: '' }
})

const confidencePercent = computed(() => Math.round(props.confidence * 100))

const progressColor = computed(() => {
  if (props.confidence >= 0.8) return '#67c23a'
  if (props.confidence >= 0.5) return '#e6a23c'
  return '#f56c6c'
})

const locationColor = computed(() => {
  // Match colors from the chart palette for consistency
  const colors = {
    'Cell membrane': '#409EFF',
    'Cytoplasm': '#E6A23C',
    'ER': '#67C23A',
    'Golgi apparatus': '#F56C6C',
    'Lysosome + Vacuole': '#909399',
    'Mitochondrion': '#E4566C',
    'Nucleus': '#8B5CF6',
    'Peroxisome': '#36CFC9',
    'Plastid': '#73D13D',
    'Extracellular': '#597EF7'
  }
  return colors[props.location] || '#409EFF'
})

const membraneType = computed(() => {
  if (props.membrane === 'Soluble') return 'success'
  if (props.membrane === 'Membrane') return 'warning'
  return 'info'
})
</script>

<style scoped>
.result-card {
  animation: fadeIn 0.5s ease;
}

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(16px); }
  to { opacity: 1; transform: translateY(0); }
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-weight: 600;
}

.result-col {
  text-align: center;
}

.result-label {
  font-size: 14px;
  color: #909399;
  margin-bottom: 12px;
}

.location-tag {
  font-size: 18px;
  padding: 8px 20px;
  border-radius: 8px;
}

.confidence-ring {
  margin-top: 16px;
  display: flex;
  justify-content: center;
}

.percentage-text {
  font-size: 22px;
  font-weight: 700;
}

.membrane-tag {
  font-size: 18px;
  padding: 8px 20px;
}

.membrane-confidence {
  margin-top: 16px;
}

.membrane-pct {
  font-size: 22px;
  font-weight: 700;
  color: #303133;
}

.membrane-label {
  font-size: 13px;
  color: #909399;
  margin-left: 4px;
}

.info-table {
  margin-top: 20px;
}

.seq-id {
  font-family: 'Courier New', monospace;
  font-size: 12px;
  color: #409eff;
}
</style>
