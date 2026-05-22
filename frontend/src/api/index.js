import axios from 'axios'

const client = axios.create({
  baseURL: '/api',
  timeout: 65000,
  headers: { 'Content-Type': 'application/json' }
})

client.interceptors.response.use(
  (res) => {
    const body = res.data
    if (body.code !== 200) {
      return Promise.reject(new Error(body.message || 'Unknown error'))
    }
    return body.data
  },
  (err) => {
    if (err.code === 'ECONNABORTED') {
      return Promise.reject(new Error('Request timeout — inference took longer than 65s'))
    }
    if (!err.response) {
      return Promise.reject(new Error('Cannot connect to backend. Please ensure the server is running on localhost:8080'))
    }
    const body = err.response.data
    const msg = (body && body.message) ? body.message : `HTTP ${err.response.status}`
    return Promise.reject(new Error(msg))
  }
)

export default client
