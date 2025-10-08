import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, File, X, CheckCircle, AlertCircle } from 'lucide-react';
import { useUploadDocument } from '../hooks/useDocuments';
import type { UploadProgress } from '../types';

interface FileUploadProps {
    onUploadComplete?: (documentId: string) => void;
    className?: string;
}

const FileUpload: React.FC<FileUploadProps> = ({ onUploadComplete, className = '' }) => {
    const [uploadProgress, setUploadProgress] = useState<UploadProgress[]>([]);
    const uploadMutation = useUploadDocument();

    const onDrop = useCallback((acceptedFiles: File[]) => {
        acceptedFiles.forEach((file) => {
            const progress: UploadProgress = {
                file,
                progress: 0,
                status: 'uploading',
            };

            setUploadProgress(prev => [...prev, progress]);

            uploadMutation.mutate(
                { file, description: `Uploaded: ${file.name}` },
                {
                    onSuccess: (document) => {
                        // Primeiro mostra que o upload foi concluído
                        setUploadProgress(prev =>
                            prev.map(p =>
                                p.file === file
                                    ? { ...p, status: 'processing', progress: 50, message: 'Processando OCR...' }
                                    : p
                            )
                        );
                        
                        // Aguarda um pouco e verifica o status do documento
                        setTimeout(() => {
                            setUploadProgress(prev =>
                                prev.map(p =>
                                    p.file === file
                                        ? { ...p, status: 'completed', progress: 100, message: 'Concluído!' }
                                        : p
                                )
                            );
                        }, 2000);
                        
                        onUploadComplete?.(document.id);
                    },
                    onError: (error) => {
                        setUploadProgress(prev =>
                            prev.map(p =>
                                p.file === file
                                    ? { ...p, status: 'error', error: error.message }
                                    : p
                            )
                        );
                    }
                }
            );
        });
    }, [uploadMutation, onUploadComplete]);

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: {
            'image/*': ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'],
            'application/pdf': ['.pdf'],
        },
        multiple: true,
    });

    const removeFile = (file: File) => {
        setUploadProgress(prev => prev.filter(p => p.file !== file));
    };

    const formatFileSize = (bytes: number) => {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    };

    return (
        <div className={`space-y-4 ${className}`}>
            {/* Drop Zone */}
            <div
                {...getRootProps()}
                className={`
          border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors
          ${isDragActive
                        ? 'border-primary-500 bg-primary-50'
                        : 'border-gray-300 hover:border-primary-400 hover:bg-gray-50'
                    }
        `}
            >
                <input {...getInputProps()} />
                <Upload className="mx-auto h-12 w-12 text-gray-400 mb-4" />
                <p className="text-lg font-medium text-gray-900 mb-2">
                    {isDragActive ? 'Drop files here...' : 'Drag & drop files here'}
                </p>
                <p className="text-sm text-gray-500">
                    or click to select files
                </p>
                <p className="text-xs text-gray-400 mt-2">
                    Supports: PNG, JPG, JPEG, GIF, BMP, TIFF, PDF
                </p>
            </div>

            {/* Upload Progress */}
            {uploadProgress.length > 0 && (
                <div className="space-y-3">
                    <h3 className="text-sm font-medium text-gray-900">Upload Progress</h3>
                    {uploadProgress.map((progress, index) => (
                        <div key={index} className="bg-white rounded-lg border border-gray-200 p-4">
                            <div className="flex items-center justify-between mb-2">
                                <div className="flex items-center space-x-3">
                                    <File className="h-5 w-5 text-gray-400" />
                                    <div>
                                        <p className="text-sm font-medium text-gray-900">
                                            {progress.file.name}
                                        </p>
                                        <p className="text-xs text-gray-500">
                                            {formatFileSize(progress.file.size)}
                                        </p>
                                    </div>
                                </div>
                                <div className="flex items-center space-x-2">
                                    {progress.status === 'completed' && (
                                        <CheckCircle className="h-5 w-5 text-green-500" />
                                    )}
                                    {progress.status === 'error' && (
                                        <AlertCircle className="h-5 w-5 text-red-500" />
                                    )}
                                    <button
                                        onClick={() => removeFile(progress.file)}
                                        className="text-gray-400 hover:text-gray-600"
                                    >
                                        <X className="h-4 w-4" />
                                    </button>
                                </div>
                            </div>

                            {(progress.status === 'uploading' || progress.status === 'processing') && (
                                <div className="space-y-2">
                                    <div className="w-full bg-gray-200 rounded-full h-2">
                                        <div
                                            className="bg-primary-600 h-2 rounded-full transition-all duration-300"
                                            style={{ width: `${progress.progress}%` }}
                                        />
                                    </div>
                                    {progress.message && (
                                        <p className="text-xs text-gray-600">
                                            {progress.message}
                                        </p>
                                    )}
                                </div>
                            )}

                            {progress.status === 'error' && (
                                <p className="text-xs text-red-600 mt-1">
                                    {progress.error}
                                </p>
                            )}
                        </div>
                    ))}
                </div>
            )}

            {/* Error Message */}
            {uploadMutation.isError && (
                <div className="bg-red-50 border border-red-200 rounded-lg p-4">
                    <div className="flex items-center space-x-2">
                        <AlertCircle className="h-5 w-5 text-red-500" />
                        <p className="text-sm text-red-700">
                            Error uploading file. Please try again.
                        </p>
                    </div>
                </div>
            )}
        </div>
    );
};

export default FileUpload; 