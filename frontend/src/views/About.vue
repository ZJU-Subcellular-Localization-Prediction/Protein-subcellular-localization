<template>
  <div class="about-page">
    <h1>About This Project</h1>

    <el-card class="about-card">
      <h2>Project Overview</h2>
      <p>
        This is a web application for predicting protein subcellular localization.
        Given an amino acid sequence, the model predicts which cellular compartment
        the protein localizes to (10 classes) and its membrane binding state (3 classes).
      </p>
    </el-card>

    <el-row :gutter="24" class="info-row">
      <el-col :span="12">
        <el-card>
          <h2>Tech Stack</h2>
          <el-table :data="techStack" stripe size="small">
            <el-table-column prop="layer" label="Layer" width="140" />
            <el-table-column prop="tech" label="Technology" />
          </el-table>
        </el-card>
      </el-col>
      <el-col :span="12">
        <el-card>
          <h2>References</h2>
          <ul class="ref-list">
            <li>Almagro Armenteros JJ, et al. <em>DeepLoc: prediction of protein subcellular localization using deep learning</em>. Bioinformatics, 2017.</li>
            <li>ESM-2: Lin Z, et al. <em>Evolutionary-scale prediction of atomic-level protein structure with a language model</em>. Science, 2023.</li>
            <li><a href="https://github.com/ClaudiaRaffaelli/Protein-subcellular-localization" target="_blank">Original Keras/TensorFlow Project</a></li>
          </ul>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup>
const techStack = [
  { layer: 'Feature Extraction', tech: 'HuggingFace ESM-2 (esm2_t30_150M_UR50D)' },
  { layer: 'Deep Learning', tech: 'PyTorch 2.x + CUDA' },
  { layer: 'Backend', tech: 'Java Spring Boot 3.x + MyBatis-Plus' },
  { layer: 'Database', tech: 'MySQL 8.x' },
  { layer: 'Frontend', tech: 'Vue 3 + Element Plus + ECharts' },
]
</script>

<style scoped>
.about-page {
  max-width: 900px;
  margin: 0 auto;
}

.about-page h1 {
  font-size: 24px;
  margin-bottom: 24px;
}

.about-card {
  margin-bottom: 24px;
}

.about-card h2, .info-row h2 {
  font-size: 18px;
  margin-bottom: 12px;
  color: #303133;
}

.about-card p {
  font-size: 14px;
  color: #606266;
  line-height: 1.8;
}

.info-row {
  margin-bottom: 24px;
}

.ref-list {
  padding-left: 20px;
  font-size: 13px;
  color: #606266;
  line-height: 1.8;
}

.ref-list li {
  margin-bottom: 8px;
}
</style>
