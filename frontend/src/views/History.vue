<template>
  <div class="history-page">
    <el-page-header @back="$router.push('/')" title="Back">
      <template #content>
        <span class="page-title">Prediction History</span>
      </template>
      <template #extra>
        <el-button type="primary" size="small" @click="$router.push('/predict')">
          <el-icon><Plus /></el-icon>
          New Prediction
        </el-button>
      </template>
    </el-page-header>

    <HistoryTable />
  </div>
</template>

<script setup>
import HistoryTable from '@/components/HistoryTable.vue'
</script>

<style scoped>
.history-page {
  max-width: 1000px;
  margin: 0 auto;
}

.page-title {
  font-size: 18px;
  font-weight: 600;
}
</style>
