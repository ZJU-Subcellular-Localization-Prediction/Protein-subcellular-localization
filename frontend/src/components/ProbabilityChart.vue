<template>
  <el-card class="chart-card" shadow="hover">
    <template #header>
      <span class="chart-title">Location Probability Distribution</span>
    </template>

    <v-chart
      v-if="hasData"
      :option="chartOption"
      :autoresize="true"
      class="chart-instance"
    />
    <el-empty v-else description="No probability data available" />
  </el-card>
</template>

<script setup>
import { computed } from 'vue'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { BarChart } from 'echarts/charts'
import { GridComponent, TooltipComponent } from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'

use([BarChart, GridComponent, TooltipComponent, CanvasRenderer])

const props = defineProps({
  probabilities: { type: Object, default: () => ({}) }
})

// Fixed canonical order and colors matching the cell diagram palette
const CATEGORIES = [
  'Cell membrane', 'Cytoplasm', 'ER', 'Golgi apparatus',
  'Lysosome + Vacuole', 'Mitochondrion', 'Nucleus',
  'Peroxisome', 'Plastid', 'Extracellular'
]

const COLORS = [
  '#409EFF', '#E6A23C', '#67C23A', '#F56C6C', '#909399',
  '#E4566C', '#8B5CF6', '#36CFC9', '#73D13D', '#597EF7'
]

const CATEGORY_SHORT = [
  'Membrane', 'Cytoplasm', 'ER', 'Golgi', 'Lyso+Vac',
  'Mito', 'Nucleus', 'Peroxi', 'Plastid', 'Extra'
]

const hasData = computed(() => {
  return props.probabilities && Object.keys(props.probabilities).length > 0
})

const chartOption = computed(() => {
  const values = CATEGORIES.map(k => props.probabilities[k] || 0)
  const maxIdx = values.indexOf(Math.max(...values))

  return {
    tooltip: {
      trigger: 'axis',
      axisPointer: { type: 'shadow' },
      formatter: (params) => {
        const p = params[0]
        return `${CATEGORIES[p.dataIndex]}<br/>Probability: <b>${(p.value * 100).toFixed(2)}%</b>`
      }
    },
    grid: { left: 100, right: 40, top: 10, bottom: 20 },
    xAxis: {
      type: 'value',
      min: 0,
      max: 1,
      axisLabel: { formatter: '{value}' },
      splitLine: { lineStyle: { type: 'dashed', color: '#e4e7ed' } }
    },
    yAxis: {
      type: 'category',
      data: CATEGORY_SHORT,
      axisTick: { show: false },
      axisLine: { show: false },
      axisLabel: { fontSize: 12, color: '#606266' }
    },
    series: [{
      type: 'bar',
      data: values.map((v, i) => ({
        value: v,
        itemStyle: {
          color: COLORS[i],
          borderRadius: [0, 4, 4, 0],
          borderWidth: i === maxIdx ? 3 : 0,
          borderColor: i === maxIdx ? '#303133' : undefined
        }
      })),
      barWidth: 16,
      emphasis: {
        itemStyle: { shadowBlur: 8, shadowColor: 'rgba(0,0,0,0.2)' }
      },
      label: {
        show: true,
        position: 'right',
        formatter: (p) => (p.value * 100).toFixed(1) + '%',
        fontSize: 11,
        color: '#909399'
      }
    }]
  }
})
</script>

<style scoped>
.chart-card {
  animation: fadeIn 0.6s ease;
}

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(16px); }
  to { opacity: 1; transform: translateY(0); }
}

.chart-title {
  font-weight: 600;
}

.chart-instance {
  width: 100%;
  height: 340px;
}
</style>
