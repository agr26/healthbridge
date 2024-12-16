const API_BASE_URL = 'http://localhost:8000/api';

const handleResponse = async (response) => {
  if (!response.ok) {
    const error = await response.json().catch(() => ({
      message: 'An error occurred'
    }));
    throw new Error(error.message || 'Request failed');
  }
  return response.json();
};

export const patientService = {
  async getPatient(patientId) {
    const response = await fetch(`${API_BASE_URL}/patient/${patientId}`);
    return handleResponse(response);
  },

  async getRiskAssessment(patientId) {
    const response = await fetch(`${API_BASE_URL}/patient/${patientId}/risk`);
    return handleResponse(response);
  },

  async getVitals(patientId) {
    const response = await fetch(`${API_BASE_URL}/patient/${patientId}/vitals`);
    return handleResponse(response);
  },

  async getLabs(patientId) {
    const response = await fetch(`${API_BASE_URL}/patient/${patientId}/labs`);
    return handleResponse(response);
  },

  async getAlerts(patientId) {
    const response = await fetch(`${API_BASE_URL}/patient/${patientId}/alerts`);
    return handleResponse(response);
  }
};

export const analyticsService = {
  async getReadmissionAnalytics() {
    const response = await fetch(`${API_BASE_URL}/analytics/readmissions`);
    return handleResponse(response);
  },

  async getQualityMetrics() {
    const response = await fetch(`${API_BASE_URL}/analytics/quality`);
    return handleResponse(response);
  }
};