import client from './index'

export function postPredict(sequence) {
  return client.post('/predict', { sequence })
}

export function getHistory(page = 1, size = 20) {
  return client.get('/history', { params: { page, size } })
}

export function getHistoryById(id) {
  return client.get(`/history/${id}`)
}
