import axios from 'axios';
import type { AxiosInstance } from 'axios';
import type {
    Document,
    AICorrection,
    AIModel,
    Statistics,
    ApiResponse,
    PaginatedResponse
} from '../types';

// API Configuration
const API_BASE_URL = import.meta.env.VITE_API_URL || '/api';

class ApiService {
    private api: AxiosInstance;

    constructor() {
        this.api = axios.create({
            baseURL: API_BASE_URL,
            timeout: 30000,
            headers: {
                'Content-Type': 'application/json',
            },
        });

        // Request interceptor
        this.api.interceptors.request.use(
            (config) => {
                // Add auth token if available
                const token = localStorage.getItem('auth_token');
                if (token) {
                    config.headers.Authorization = `Bearer ${token}`;
                }
                return config;
            },
            (error) => {
                return Promise.reject(error);
            }
        );

        // Response interceptor
        this.api.interceptors.response.use(
            (response) => response,
            (error) => {
                if (error.response?.status === 401) {
                    // Handle unauthorized access
                    localStorage.removeItem('auth_token');
                    window.location.href = '/login';
                }
                return Promise.reject(error);
            }
        );
    }

    // Health check
    async healthCheck(): Promise<ApiResponse<{ status: string }>> {
        const response = await this.api.get('/health');
        return response.data;
    }

    // Documents
    async getDocuments(page = 1, limit = 10): Promise<PaginatedResponse<Document>> {
        const response = await this.api.get(`/documents/?page=${page}&limit=${limit}`);
        return response.data;
    }

    async getDocument(id: string): Promise<Document> {
        const response = await this.api.get(`/documents/${id}/`);
        return response.data;
    }

    async uploadDocument(file: File, description?: string): Promise<Document> {
        const formData = new FormData();
        formData.append('file', file);
        if (description) {
            formData.append('description', description);
        }

        const response = await this.api.post('/documents/upload', formData, {
            headers: {
                'Content-Type': 'multipart/form-data',
            },
        });
        return response.data;
    }

    async deleteDocument(id: string): Promise<void> {
        await this.api.delete(`/documents/${id}`);
    }

    // AI Corrections
    async getAICorrections(documentId: string): Promise<AICorrection[]> {
        const response = await this.api.get(`/documents/${documentId}/corrections`);
        return response.data;
    }

    async createAICorrection(documentId: string): Promise<AICorrection> {
        const response = await this.api.post(`/documents/${documentId}/corrections`);
        return response.data;
    }

    async updateCorrection(correctionId: string, corrections: any[]): Promise<AICorrection> {
        const response = await this.api.put(`/corrections/${correctionId}`, { corrections });
        return response.data;
    }

    // AI Model
    async getAIModel(): Promise<AIModel> {
        const response = await this.api.get('/ai/model');
        return response.data;
    }

    async trainAIModel(): Promise<{ message: string }> {
        const response = await this.api.post('/ai/train');
        return response.data;
    }

    async getAIStatus(): Promise<{ status: string; accuracy: number }> {
        const response = await this.api.get('/ai/status');
        return response.data;
    }

    // Statistics
    async getStatistics(): Promise<Statistics> {
        const response = await this.api.get('/statistics');
        return response.data;
    }

    // File upload with progress
    async uploadWithProgress(
        file: File,
        onProgress?: (progress: number) => void,
        description?: string
    ): Promise<Document> {
        const formData = new FormData();
        formData.append('file', file);
        if (description) {
            formData.append('description', description);
        }

        const response = await this.api.post('/documents/upload', formData, {
            headers: {
                'Content-Type': 'multipart/form-data',
            },
            onUploadProgress: (progressEvent) => {
                if (progressEvent.total && onProgress) {
                    const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
                    onProgress(progress);
                }
            },
        });
        return response.data;
    }

    async processDocument(id: string): Promise<any> {
        const response = await this.api.post(`/documents/${id}/process`);
        return response.data;
    }
}

export const apiService = new ApiService();
export default apiService; 