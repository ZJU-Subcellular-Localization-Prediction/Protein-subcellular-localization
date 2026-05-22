<template>
  <el-card class="cell-diagram-card" shadow="hover">
    <template #header>
      <span class="card-title">Cell Diagram</span>
    </template>

    <div class="diagram-wrapper">
      <svg viewBox="0 0 400 400" class="cell-svg" xmlns="http://www.w3.org/2000/svg">

        <!-- Cytoplasm background -->
        <ellipse
          cx="200" cy="200" rx="166" ry="136"
          fill="#FFF9E6" stroke="none"
          class="compartment"
          :class="{ active: highlightLocation === 'Cytoplasm' }"
          data-location="Cytoplasm"
        />

        <!-- Cell membrane (outer boundary) -->
        <ellipse
          cx="200" cy="200" rx="170" ry="140"
          fill="none"
          class="compartment membrane-outline"
          :class="{ active: highlightLocation === 'Cell membrane' }"
          :style="membraneStyle"
          data-location="Cell membrane"
        />

        <!-- Nucleus -->
        <g
          class="compartment"
          :class="{ active: highlightLocation === 'Nucleus' }"
          data-location="Nucleus"
        >
          <circle cx="170" cy="180" r="45" :fill="nucleusFill" stroke="none" />
          <circle cx="175" cy="175" r="14" fill="#C4B5FD" opacity="0.8" />
        </g>

        <!-- Rough ER (around nucleus) -->
        <g
          class="compartment"
          :class="{ active: highlightLocation === 'ER' }"
          data-location="ER"
        >
          <path
            d="M217,170 C240,155 258,168 252,195 C248,215 235,210 230,200 C225,190 220,180 217,170Z"
            :fill="erFill"
            stroke="none"
            opacity="0.8"
          />
          <path
            d="M260,200 C275,195 282,210 275,225 C265,240 255,235 252,220"
            :fill="erFill"
            stroke="none"
            opacity="0.6"
          />
        </g>

        <!-- Golgi apparatus (stacked arcs right of nucleus) -->
        <g
          class="compartment"
          :class="{ active: highlightLocation === 'Golgi apparatus' }"
          data-location="Golgi apparatus"
        >
          <path d="M120,95 C145,80 175,85 195,95" :stroke="golgiColor" stroke-width="3" fill="none" />
          <path d="M118,105 C143,90 177,95 197,105" :stroke="golgiColor" stroke-width="3" fill="none" />
          <path d="M122,115 C145,100 175,105 193,115" :stroke="golgiColor" stroke-width="3" fill="none" />
          <path d="M128,125 C148,112 172,115 187,125" :stroke="golgiColor" stroke-width="3" fill="none" />
        </g>

        <!-- Mitochondria (3 scattered) -->
        <g
          v-for="(m, idx) in mitochondria"
          :key="'mito-' + idx"
          class="compartment"
          :class="{ active: highlightLocation === 'Mitochondrion' }"
          data-location="Mitochondrion"
        >
          <ellipse
            :cx="m.cx" :cy="m.cy" :rx="m.rx" :ry="m.ry"
            :transform="m.rot"
            :fill="mitoFill"
            stroke="none"
          />
          <path
            :d="m.path"
            :stroke="mitoCristaColor"
            stroke-width="1.2"
            fill="none"
            opacity="0.7"
          />
        </g>

        <!-- Lysosome / Vacuole -->
        <circle
          cx="280" cy="130" r="14"
          class="compartment"
          :class="{ active: highlightLocation === 'Lysosome + Vacuole' }"
          :fill="lysoFill"
          stroke="none"
          data-location="Lysosome + Vacuole"
        />

        <!-- Peroxisome -->
        <circle
          cx="100" cy="260" r="9"
          class="compartment"
          :class="{ active: highlightLocation === 'Peroxisome' }"
          :fill="peroxiFill"
          stroke="none"
          data-location="Peroxisome"
        />

        <!-- Plastid (dashed, plant-like) -->
        <g
          class="compartment"
          :class="{ active: highlightLocation === 'Plastid' }"
          data-location="Plastid"
        >
          <ellipse cx="290" cy="280" rx="22" ry="14"
            :stroke="plastidColor" stroke-width="2"
            stroke-dasharray="6,3" fill="none" />
          <line x1="275" y1="280" x2="305" y2="280"
            :stroke="plastidColor" stroke-width="1" opacity="0.4" />
        </g>

        <!-- Extracellular label -->
        <text
          x="350" y="70"
          font-size="11" font-weight="600"
          :fill="highlightLocation === 'Extracellular' ? '#597EF7' : '#909399'"
          class="compartment"
          :class="{ active: highlightLocation === 'Extracellular' }"
          data-location="Extracellular"
        >Extracellular</text>

      </svg>

      <!-- Legend -->
      <div class="color-legend">
        <span
          v-for="loc in locations"
          :key="loc.name"
          class="legend-item"
          :class="{ active: highlightLocation === loc.name }"
          @mouseenter="hoveredLocation = loc.name"
          @mouseleave="hoveredLocation = null"
        >
          <span class="legend-dot" :style="{ background: loc.color }"></span>
          {{ loc.short }}
        </span>
      </div>
    </div>
  </el-card>
</template>

<script setup>
import { ref, computed } from 'vue'

const props = defineProps({
  highlightLocation: { type: String, default: '' },
  allProbabilities: { type: Object, default: () => ({}) }
})

const hoveredLocation = ref(null)

// Color palette
const LOC_COLORS = {
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

const locations = [
  { name: 'Cell membrane', short: 'Membrane', color: '#409EFF' },
  { name: 'Cytoplasm', short: 'Cytoplasm', color: '#E6A23C' },
  { name: 'Nucleus', short: 'Nucleus', color: '#8B5CF6' },
  { name: 'ER', short: 'ER', color: '#67C23A' },
  { name: 'Golgi apparatus', short: 'Golgi', color: '#F56C6C' },
  { name: 'Mitochondrion', short: 'Mito', color: '#E4566C' },
  { name: 'Lysosome + Vacuole', short: 'Lyso+Vac', color: '#909399' },
  { name: 'Peroxisome', short: 'Peroxi', color: '#36CFC9' },
  { name: 'Plastid', short: 'Plastid', color: '#73D13D' },
  { name: 'Extracellular', short: 'Extra', color: '#597EF7' }
]

// Highlight color for membrane
const activeColor = computed(() => LOC_COLORS[props.highlightLocation] || '#DCDFE6')
const inactiveColor = '#DCDFE6'

const membraneStyle = computed(() => ({
  stroke: props.highlightLocation === 'Cell membrane' ? activeColor.value : '#DCDFE6'
}))

const nucleusFill = computed(() =>
  props.highlightLocation === 'Nucleus' ? '#DDD6FE' : '#EDE9FE'
)

const erFill = computed(() =>
  props.highlightLocation === 'ER' ? '#B9F0D3' : '#D1FAE5'
)

const golgiColor = computed(() =>
  props.highlightLocation === 'Golgi apparatus' ? '#F56C6C' : '#DCDFE6'
)

const mitoFill = computed(() =>
  props.highlightLocation === 'Mitochondrion' ? '#FECACA' : '#FEE2E2'
)

const mitoCristaColor = computed(() =>
  props.highlightLocation === 'Mitochondrion' ? '#E4566C' : '#DCDFE6'
)

const lysoFill = computed(() =>
  props.highlightLocation === 'Lysosome + Vacuole' ? '#D1D5DB' : '#E5E7EB'
)

const peroxiFill = computed(() =>
  props.highlightLocation === 'Peroxisome' ? '#A5F3FC' : '#CFFAFE'
)

const plastidColor = computed(() =>
  props.highlightLocation === 'Plastid' ? '#73D13D' : '#DCDFE6'
)

// Mitochondria data
const mitochondria = [
  { cx: 105, cy: 310, rx: 22, ry: 9, rot: 'rotate(-15, 105, 310)', path: 'M88,310 Q105,306 125,310' },
  { cx: 260, cy: 320, rx: 28, ry: 10, rot: 'rotate(10, 260, 320)', path: 'M238,320 Q260,315 285,320' },
  { cx: 310, cy: 190, rx: 20, ry: 8, rot: 'rotate(-30, 310, 190)', path: 'M295,190 Q310,186 328,190' }
]
</script>

<style scoped>
.cell-diagram-card {
  height: 100%;
  animation: fadeIn 0.7s ease;
}

@keyframes fadeIn {
  from { opacity: 0; transform: translateY(16px); }
  to { opacity: 1; transform: translateY(0); }
}

.card-title {
  font-weight: 600;
}

.diagram-wrapper {
  width: 100%;
}

.cell-svg {
  width: 100%;
  height: auto;
  max-height: 380px;
}

.compartment {
  transition: fill 0.5s ease, stroke 0.5s ease, opacity 0.5s ease;
  cursor: pointer;
}

.compartment:not(.active) {
  opacity: 0.65;
}

.compartment.active {
  opacity: 1;
}

.membrane-outline {
  stroke-width: 4;
  transition: stroke 0.5s ease, stroke-width 0.5s ease;
}

.membrane-outline.active {
  stroke-width: 6;
  filter: drop-shadow(0 0 6px currentColor);
}

.color-legend {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 12px;
  padding-top: 12px;
  border-top: 1px solid #ebeef5;
  justify-content: center;
}

.legend-item {
  display: flex;
  align-items: center;
  gap: 3px;
  font-size: 11px;
  color: #909399;
  cursor: pointer;
  padding: 2px 6px;
  border-radius: 4px;
  transition: all 0.2s ease;
}

.legend-item.active {
  color: #303133;
  font-weight: 600;
  background: #f0f2f5;
}

.legend-dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  display: inline-block;
  flex-shrink: 0;
}
</style>
