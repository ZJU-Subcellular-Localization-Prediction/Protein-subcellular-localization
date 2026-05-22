<template>
  <div class="home">
    <div class="hero">
      <h1 class="hero-title">Protein Subcellular Localization Prediction</h1>
      <p class="hero-subtitle">
        Deep learning powered prediction of protein subcellular locations using
        ESM-2 embeddings and CNN + BiLSTM + Attention architecture
      </p>
      <el-button type="primary" size="large" @click="$router.push('/predict')">
        Try Now
        <el-icon class="el-icon--right"><ArrowRight /></el-icon>
      </el-button>
    </div>

    <el-row :gutter="24" class="tech-stack">
      <el-col :span="6" v-for="item in techItems" :key="item.title">
        <el-card shadow="hover" class="tech-card">
          <el-icon :size="32" :color="item.color">
            <component :is="item.icon" />
          </el-icon>
          <h3>{{ item.title }}</h3>
          <p>{{ item.desc }}</p>
        </el-card>
      </el-col>
    </el-row>

    <div class="architecture">
      <h2>Model Architecture</h2>
      <p class="arch-desc">
        6 parallel Conv1d layers (kernel=1,3,5,9,15,21) → BiLSTM → Bahdanau Attention
      </p>
      <el-steps :active="3" align-center class="arch-steps">
        <el-step title="Feature Extraction" description="ESM-2 (150M params, 640-dim)" />
        <el-step title="Sequence Encoding" description="6× Multi-scale CNN + BiLSTM" />
        <el-step title="Attention" description="Bahdanau Attention Mechanism" />
        <el-step title="Prediction" description="10-class Location + 3-class Membrane" />
      </el-steps>
    </div>
  </div>
</template>

<script setup>
import { Cpu, DataAnalysis, Connection, TrendCharts } from '@element-plus/icons-vue'

const techItems = [
  { title: 'ESM-2', desc: '150M parameter protein language model for 640-dim embeddings', icon: 'Cpu', color: '#409EFF' },
  { title: 'PyTorch', desc: 'CNN + BiLSTM + Bahdanau Attention deep learning framework', icon: 'DataAnalysis', color: '#E4566C' },
  { title: 'Spring Boot', desc: 'Java backend with MyBatis-Plus + MySQL persistence', icon: 'Connection', color: '#67C23A' },
  { title: 'Vue 3', desc: 'Element Plus + ECharts interactive visualization frontend', icon: 'TrendCharts', color: '#8B5CF6' }
]
</script>

<style scoped>
.home {
  padding-top: 24px;
}

.hero {
  text-align: center;
  padding: 60px 20px 48px;
}

.hero-title {
  font-size: 32px;
  font-weight: 700;
  color: #303133;
  margin-bottom: 16px;
}

.hero-subtitle {
  font-size: 16px;
  color: #606266;
  max-width: 640px;
  margin: 0 auto 32px;
  line-height: 1.6;
}

.tech-stack {
  margin: 32px 0 48px;
}

.tech-card {
  text-align: center;
  padding: 16px 0;
  height: 180px;
}

.tech-card h3 {
  margin: 12px 0 8px;
  font-size: 16px;
  color: #303133;
}

.tech-card p {
  font-size: 13px;
  color: #909399;
  line-height: 1.5;
}

.architecture {
  text-align: center;
  padding: 32px 0 48px;
}

.architecture h2 {
  font-size: 22px;
  color: #303133;
  margin-bottom: 8px;
}

.arch-desc {
  font-size: 14px;
  color: #606266;
  margin-bottom: 32px;
}

.arch-steps {
  max-width: 800px;
  margin: 0 auto;
}
</style>
