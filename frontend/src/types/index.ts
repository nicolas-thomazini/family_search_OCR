// Document types
export interface Document {
    id: string;
    filename: string;
    original_text?: string;
    corrected_text?: string;
    confidence_score?: number;
    status: 'processing' | 'completed' | 'error' | 'uploaded';
    created_at?: string;
    updated_at?: string;
    file_size?: number;
    file_type?: string;
    upload_date?: string;
}

// AI Correction types
export interface AICorrection {
    id: string;
    document_id: string;
    original_text: string;
    corrected_text: string;
    confidence_score: number;
    corrections: Correction[];
    created_at: string;
}

export interface Correction {
    id: string;
    original: string;
    corrected: string;
    confidence: number;
    type: 'spelling' | 'grammar' | 'context' | 'format';
    position: {
        start: number;
        end: number;
    };
}

// AI Model types
export interface AIModel {
    id: string;
    name: string;
    version: string;
    status: 'active' | 'training' | 'inactive';
    accuracy: number;
    last_trained: string;
    training_data_size: number;
}

// API Response types
export interface ApiResponse<T> {
    success: boolean;
    data?: T;
    message?: string;
    error?: string;
}

export interface PaginatedResponse<T> {
    items: T[];
    total: number;
    page: number;
    per_page: number;
    total_pages: number;
}

// Upload types
export interface UploadProgress {
    file: File;
    progress: number;
    status: 'uploading' | 'processing' | 'completed' | 'error';
    error?: string;
    message?: string;
}

// Statistics types
export interface Statistics {
    total_documents: number;
    processed_documents: number;
    average_confidence: number;
    total_corrections: number;
    documents_today: number;
    ai_accuracy: number;
}

// Form types
export interface DocumentUploadForm {
    files: File[];
    description?: string;
    tags?: string[];
}

export interface CorrectionForm {
    document_id: string;
    corrections: Correction[];
} 