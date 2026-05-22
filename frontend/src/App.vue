<template>
  <el-container class="app-container">
    <el-header class="app-header">
      <div class="logo" @click="$router.push('/')">
        <el-icon :size="24"><House /></el-icon>
        <span class="logo-text">Protein Localization</span>
      </div>
      <el-menu
        mode="horizontal"
        :default-active="activeRoute"
        :ellipsis="false"
        router
        class="nav-menu"
      >
        <el-menu-item index="/">
          <el-icon><HomeFilled /></el-icon>
          <span>Home</span>
        </el-menu-item>
        <el-menu-item index="/predict">
          <el-icon><Search /></el-icon>
          <span>Predict</span>
        </el-menu-item>
        <el-menu-item index="/history">
          <el-icon><Clock /></el-icon>
          <span>History</span>
        </el-menu-item>
        <el-menu-item index="/about">
          <el-icon><User /></el-icon>
          <span>About</span>
        </el-menu-item>
      </el-menu>
    </el-header>

    <el-main class="app-main">
      <router-view />
    </el-main>

    <el-footer class="app-footer">
      <span>Protein Subcellular Localization Prediction &copy; 2026</span>
      <span class="footer-divider">|</span>
      <span>CS + Medicine Interdisciplinary Team</span>
    </el-footer>
  </el-container>
</template>

<script setup>
import { computed } from 'vue'
import { useRoute } from 'vue-router'

const route = useRoute()
const activeRoute = computed(() => route.path)
</script>

<style>
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: 'Helvetica Neue', Helvetica, 'PingFang SC', 'Microsoft YaHei', Arial, sans-serif;
  background: #f5f7fa;
  color: #303133;
}

.app-container {
  min-height: 100vh;
}

.app-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  background: #fff;
  border-bottom: 1px solid #e4e7ed;
  padding: 0 24px;
  height: 60px;
  box-shadow: 0 1px 4px rgba(0, 0, 0, 0.06);
}

.logo {
  display: flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
  color: #409eff;
  user-select: none;
}

.logo-text {
  font-size: 18px;
  font-weight: 700;
  letter-spacing: 0.5px;
  white-space: nowrap;
}

.nav-menu {
  border-bottom: none !important;
}

.nav-menu .el-menu-item {
  height: 60px;
  line-height: 60px;
}

.app-main {
  min-height: calc(100vh - 120px);
  padding: 24px;
  max-width: 1200px;
  margin: 0 auto;
  width: 100%;
}

.app-footer {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  height: 60px;
  background: #fff;
  border-top: 1px solid #e4e7ed;
  font-size: 13px;
  color: #909399;
}

.footer-divider {
  color: #dcdfe6;
}
</style>
