<template>
  <div class="sequence-input">
    <el-input
      v-model="sequence"
      type="textarea"
      :rows="6"
      placeholder="Enter an amino acid sequence (e.g. MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHL...)"
      :class="{ 'has-error': validationError }"
      @input="onInput"
    />

    <div class="input-actions">
      <div class="left-actions">
        <el-upload
          :auto-upload="false"
          :show-file-list="false"
          accept=".fasta,.txt,.fa"
          @change="onFileChange"
        >
          <el-button type="default">
            <el-icon><Upload /></el-icon>
            Upload FASTA
          </el-button>
        </el-upload>
        <el-button @click="loadExample">
          <el-icon><Document /></el-icon>
          Load Example
        </el-button>
        <el-button @click="clearAll" :disabled="!sequence">
          <el-icon><Delete /></el-icon>
          Clear
        </el-button>
      </div>
      <div class="right-actions">
        <span class="char-count">{{ validCharCount }} valid AA</span>
        <el-button type="primary" :disabled="!canSubmit" @click="submit">
          <el-icon><Search /></el-icon>
          Submit &amp; Predict
        </el-button>
      </div>
    </div>

    <el-alert
      v-if="validationError"
      :title="validationError"
      type="warning"
      :closable="false"
      show-icon
      class="validation-alert"
    />

    <el-alert
      v-if="lengthWarning"
      :title="lengthWarning"
      type="info"
      :closable="true"
      show-icon
      class="validation-alert"
    />

    <el-alert
      v-if="warning"
      :title="warning"
      type="info"
      :closable="true"
      show-icon
      class="validation-alert"
      @close="warning = ''"
    />
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'

const emit = defineEmits(['submit', 'clear'])

// ---- state ----
const sequence = ref('')
const validationError = ref('')
const warning = ref('')

// ---- example sequence (Human Insulin precursor) ----
const EXAMPLE = `MALWMRLLPLLALLALWGPDPAAAFVNQHLCGSHLVEALYLVCGERGFFYTPKTRREAED
LQVGQVELGGGPGAGSLQPLALEGSLQKRGIVEQCCTSICSLYQLENYCN`

// ---- computed ----
const validCharCount = computed(() => {
  const cleaned = sequence.value.replace(/[\s\n\r]/g, '')
  const valid = cleaned.match(/[ACDEFGHIKLMNPQRSTVWY]/gi)
  return valid ? valid.length : 0
})

const MAX_SEQ_LEN = 1000

const canSubmit = computed(() => validCharCount.value > 0)

const lengthWarning = computed(() => {
  if (validCharCount.value > MAX_SEQ_LEN) {
    return `Sequence is ${validCharCount.value} AA long — it will be center-truncated to ${MAX_SEQ_LEN} AA before inference.`
  }
  return ''
})

// ---- methods ----
function onInput() {
  // Live validation: only warn for clearly invalid characters
  const invalid = sequence.value.match(/[^ACDEFGHIKLMNPQRSTVWY\s\n\r]/gi)
  if (invalid && invalid.length > 0) {
    const unique = [...new Set(invalid.map(c => c.toUpperCase()))].join(', ')
    validationError.value = `Invalid characters detected: ${unique}. These will be removed before inference.`
  } else {
    validationError.value = ''
  }
}

function loadExample() {
  sequence.value = EXAMPLE.replace(/\s+/g, '')
  validationError.value = ''
  warning.value = 'Example loaded: Human Insulin precursor (110 amino acids)'
}

function clearAll() {
  sequence.value = ''
  validationError.value = ''
  warning.value = ''
  emit('clear')
}

function submit() {
  if (!canSubmit.value) return
  emit('submit', sequence.value)
}

function onFileChange(uploadFile) {
  const file = uploadFile.raw
  if (!file) return

  const reader = new FileReader()
  reader.onload = (e) => {
    const text = e.target.result
    // Strip FASTA header lines (starting with >)
    const lines = text.split(/[\r\n]+/)
    let seq = ''
    for (const line of lines) {
      const trimmed = line.trim()
      if (!trimmed || trimmed.startsWith('>')) continue
      seq += trimmed
    }
    sequence.value = seq
    validationError.value = ''
    warning.value = `Loaded from file: ${file.name} (${seq.length} characters)`
    onInput()
  }
  reader.onerror = () => {
    warning.value = `Failed to read file: ${file.name}`
  }
  reader.readAsText(file)
}
</script>

<style scoped>
.sequence-input {
  width: 100%;
}

.sequence-input :deep(.el-textarea__inner) {
  font-family: 'Courier New', Courier, monospace;
  font-size: 13px;
  line-height: 1.6;
  letter-spacing: 0.5px;
}

.sequence-input :deep(.el-textarea__inner.has-error) {
  border-color: #e6a23c;
}

.input-actions {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: 12px;
  flex-wrap: wrap;
  gap: 8px;
}

.left-actions, .right-actions {
  display: flex;
  align-items: center;
  gap: 8px;
}

.char-count {
  font-size: 13px;
  color: #909399;
  margin-right: 8px;
}

.validation-alert {
  margin-top: 12px;
}
</style>
