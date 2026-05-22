<template>
  <div class="history-table">
    <div class="table-toolbar">
      <el-input
        v-model="searchText"
        placeholder="Search by Sequence ID..."
        clearable
        :prefix-icon="Search"
        style="width: 280px"
        @input="onSearch"
      />
      <span class="total-hint">{{ filteredRecords.length }} / {{ total }} records</span>
    </div>

    <el-table
      :data="pagedRecords"
      stripe
      highlight-current-row
      @row-click="onRowClick"
      style="width: 100%"
      v-loading="loading"
      empty-text="No prediction history yet"
    >
      <el-table-column prop="id" label="ID" width="60" />
      <el-table-column prop="sequenceId" label="Sequence ID" width="130">
        <template #default="{ row }">
          <code class="seq-id-cell">{{ row.sequenceId }}</code>
        </template>
      </el-table-column>
      <el-table-column prop="predictedLocation" label="Location" width="150">
        <template #default="{ row }">
          <el-tag
            :color="getLocationColor(row.predictedLocation)"
            size="small"
            effect="dark"
          >
            {{ row.predictedLocation }}
          </el-tag>
        </template>
      </el-table-column>
      <el-table-column prop="locationConfidence" label="Confidence" width="100" align="center">
        <template #default="{ row }">
          <span :style="{ color: getConfidenceColor(row.locationConfidence) }">
            {{ (row.locationConfidence * 100).toFixed(1) }}%
          </span>
        </template>
      </el-table-column>
      <el-table-column prop="predictedMembrane" label="Membrane" width="100">
        <template #default="{ row }">
          <el-tag v-if="row.predictedMembrane" :type="getMembraneType(row.predictedMembrane)" size="small">
            {{ row.predictedMembrane }}
          </el-tag>
          <span v-else class="na-text">—</span>
        </template>
      </el-table-column>
      <el-table-column prop="createdAt" label="Time" min-width="160">
        <template #default="{ row }">
          {{ formatTime(row.createdAt) }}
        </template>
      </el-table-column>
    </el-table>

    <el-pagination
      v-if="filteredRecords.length > 0"
      v-model:current-page="currentPage"
      :page-size="pageSize"
      :total="filteredRecords.length"
      layout="total, prev, pager, next"
      small
      class="table-pagination"
    />

    <!-- Detail dialog -->
    <el-dialog
      v-model="dialogVisible"
      title="Prediction Detail"
      width="680px"
      destroy-on-close
    >
      <div v-if="detail" class="detail-content">
        <el-descriptions :column="2" border size="small">
          <el-descriptions-item label="Record ID">{{ detail.id }}</el-descriptions-item>
          <el-descriptions-item label="Sequence ID">
            <code>{{ detail.sequenceId }}</code>
          </el-descriptions-item>
          <el-descriptions-item label="Predicted Location">
            <el-tag
              :color="getLocationColor(detail.predictedLocation)"
              size="small"
              effect="dark"
            >{{ detail.predictedLocation }}</el-tag>
          </el-descriptions-item>
          <el-descriptions-item label="Confidence">
            {{ (detail.locationConfidence * 100).toFixed(1) }}%
          </el-descriptions-item>
          <el-descriptions-item label="Membrane">{{ detail.predictedMembrane || 'N/A' }}</el-descriptions-item>
          <el-descriptions-item label="Model">{{ detail.modelVersion || 'v1' }}</el-descriptions-item>
          <el-descriptions-item label="Inference Time">{{ detail.inferenceTimeMs }} ms</el-descriptions-item>
          <el-descriptions-item label="Created At">{{ formatTime(detail.createdAt) }}</el-descriptions-item>
        </el-descriptions>

        <div v-if="detail.allProbabilities" class="detail-chart">
          <h4>Location Probability Distribution</h4>
          <ProbabilityChart :probabilities="detail.allProbabilities" />
        </div>
      </div>
      <el-empty v-else description="Loading..." />
    </el-dialog>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, watch } from 'vue'
import { Search } from '@element-plus/icons-vue'
import { getHistory, getHistoryById } from '@/api/predict'
import ProbabilityChart from '@/components/ProbabilityChart.vue'

const emit = defineEmits(['view-detail'])

// ---- state ----
const loading = ref(false)
const allRecords = ref([])
const total = ref(0)
const searchText = ref('')
const currentPage = ref(1)
const pageSize = ref(20)

const dialogVisible = ref(false)
const detail = ref(null)

// ---- computed ----
const filteredRecords = computed(() => {
  if (!searchText.value.trim()) return allRecords.value
  const q = searchText.value.trim().toLowerCase()
  return allRecords.value.filter(r => r.sequenceId && r.sequenceId.toLowerCase().includes(q))
})

const pagedRecords = computed(() => {
  const start = (currentPage.value - 1) * pageSize.value
  return filteredRecords.value.slice(start, start + pageSize.value)
})

// ---- watchers ----
watch(searchText, () => { currentPage.value = 1 })

// ---- lifecycle ----
onMounted(() => fetchHistory())

// ---- methods ----
async function fetchHistory() {
  loading.value = true
  try {
    const resp = await getHistory(1, 200) // fetch up to 200 records at once
    allRecords.value = resp.records || []
    total.value = resp.total || 0
  } catch (e) {
    // silently fall back to list data
  } finally {
    loading.value = false
  }
}

function onSearch() {
  currentPage.value = 1
}

async function onRowClick(row) {
  emit('view-detail', row.id)
  dialogVisible.value = true
  detail.value = null
  try {
    const resp = await getHistoryById(row.id)
    detail.value = resp
  } catch (e) {
    detail.value = row // fallback to list data
  }
}

// ---- helpers ----
function getLocationColor(location) {
  const colors = {
    'Cell membrane': '#409EFF', 'Cytoplasm': '#E6A23C', 'ER': '#67C23A',
    'Golgi apparatus': '#F56C6C', 'Lysosome + Vacuole': '#909399',
    'Mitochondrion': '#E4566C', 'Nucleus': '#8B5CF6',
    'Peroxisome': '#36CFC9', 'Plastid': '#73D13D', 'Extracellular': '#597EF7'
  }
  return colors[location] || '#409EFF'
}

function getConfidenceColor(conf) {
  if (conf >= 0.8) return '#67c23a'
  if (conf >= 0.5) return '#e6a23c'
  return '#f56c6c'
}

function getMembraneType(membrane) {
  if (membrane === 'Soluble') return 'success'
  if (membrane === 'Membrane') return 'warning'
  return 'info'
}

function formatTime(t) {
  if (!t) return ''
  if (typeof t === 'string') return t.replace('T', ' ')
  return t
}
</script>

<style scoped>
.history-table {
  margin-top: 24px;
}

.table-toolbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}

.total-hint {
  font-size: 13px;
  color: #909399;
}

.seq-id-cell {
  font-family: 'Courier New', monospace;
  font-size: 12px;
  color: #409eff;
}

.na-text {
  color: #c0c4cc;
}

.table-pagination {
  margin-top: 16px;
  justify-content: flex-end;
}

.detail-content {
  max-height: 70vh;
  overflow-y: auto;
}

.detail-chart {
  margin-top: 20px;
}

.detail-chart h4 {
  font-size: 14px;
  color: #606266;
  margin-bottom: 8px;
}
</style>
