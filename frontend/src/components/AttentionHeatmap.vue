<template>
  <el-card class="heatmap-card" shadow="hover">
    <template #header>
      <div class="card-header">
        <span class="card-title">Attention Weights</span>
        <span class="card-hint">Drag the slider to explore the sequence</span>
      </div>
    </template>

    <v-chart
      v-if="hasData"
      :option="chartOption"
      :autoresize="true"
      class="heatmap-chart"
    />
    <el-empty v-else description="Attention weights not available" />
  </el-card>
</template>

<script setup>
import { computed } from 'vue'
import VChart from 'vue-echarts'
import { use } from 'echarts/core'
import { HeatmapChart } from 'echarts/charts'
import { GridComponent, VisualMapComponent, DataZoomComponent, TooltipComponent } from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'

use([HeatmapChart, GridComponent, VisualMapComponent, DataZoomComponent, TooltipComponent, CanvasRenderer])

const props = defineProps({
  attentionWeights: { type: Array, default: () => [] }
})

const hasData = computed(() => {
  return props.attentionWeights && props.attentionWeights.length > 0
})

const chartOption = computed(() => {
  // Flatten 2D array: attentionWeights is number[][]
  // Each row: positions, each col (usually 1): weight value
  const weights = props.attentionWeights || []
  const flat = Array.isArray(weights[0])
    ? weights[0].length !== undefined ? weights[0] : []
    : weights

  const data = flat.map((w, i) => [i, 0, w])
  const maxVal = Math.max(...flat, 0.001)

  return {
    tooltip: {
      position: 'top',
      formatter: (p) => `Position ${p.data[0] + 1}<br/>Weight: <b>${p.data[2].toFixed(6)}</b>`
    },
    grid: { top: 10, bottom: 70, left: 50, right: 20 },
    xAxis: {
      type: 'category',
      data: flat.map((_, i) => (i + 1) % 100 === 1 ? i + 1 : ''),
      axisLabel: { interval: 0, fontSize: 10, rotate: 0 },
      splitArea: { show: true }
    },
    yAxis: {
      type: 'category',
      data: [''],
      axisLabel: { show: false },
      axisTick: { show: false }
    },
    visualMap: {
      min: 0,
      max: maxVal,
      calculable: true,
      orient: 'horizontal',
      left: 'center',
      bottom: 0,
      inRange: {
        color: ['#eff6ff', '#93c5fd', '#3b82f6', '#1e40af', '#991b1b']
      }
    },
    series: [{
      type: 'heatmap',
      data: data,
      itemStyle: { borderWidth: 0.5, borderColor: '#fff' },
      label: { show: false },
      emphasis: {
        itemStyle: {
          shadowBlur: 8,
          shadowColor: 'rgba(0,0,0,0.3)'
        }
      }
    }],
    dataZoom: [
      {
        type: 'slider',
        xAxisIndex: 0,
        bottom: 35,
        height: 20,
        start: 0,
        end: 100
      },
      {
        type: 'inside',
        xAxisIndex: 0
      }
    ]
  }
})
</script>

<style scoped>
.heatmap-card {
  margin-top: 20px;
  animation: fadeIn 0.8s ease;
}

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(16px); }
  to { opacity: 1; transform: translateY(0); }
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.card-title {
  font-weight: 600;
}

.card-hint {
  font-size: 12px;
  color: #909399;
}

.heatmap-chart {
  width: 100%;
  height: 180px;
}
</style>
