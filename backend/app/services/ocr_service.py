import pytesseract
import cv2
import numpy as np
from PIL import Image
import os
from typing import Tuple, Dict, Any
from app.config import settings
import logging
from skimage import exposure, transform
import hashlib
import time

logger = logging.getLogger(__name__)

class OCRService:
    def __init__(self):
        # Configurar Tesseract
        pytesseract.pytesseract.tesseract_cmd = settings.TESSERACT_CMD
        
    def preprocess_image(self, image_path: str, save_debug: bool = True, force_reprocess: bool = False) -> str:
        """
        Pré-processa a imagem usando múltiplas versões e seleciona a melhor.
        Retorna o caminho da melhor imagem pré-processada.
        """
        import cv2
        import numpy as np
        import os
        import hashlib
        import time
        from skimage.transform import rotate
        
        # Gerar hash único baseado no timestamp para evitar cache
        timestamp = str(time.time())
        file_hash = hashlib.md5(f"{image_path}_{timestamp}".encode()).hexdigest()[:8]
        
        # Ler imagem
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Não foi possível ler a imagem: {image_path}")
        
        # 1. Converter para escala de cinza
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 2. Remover ruído
        denoised = cv2.fastNlMeansDenoising(gray, h=15)
        
        # Criar múltiplas versões com diferentes parâmetros
        versions = []
        
        # Versão 1: CLAHE suave + binarização adaptativa
        clahe1 = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
        contrast1 = clahe1.apply(denoised)
        binary1 = cv2.adaptiveThreshold(contrast1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)
        sharp1 = cv2.filter2D(binary1, -1, np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]]))
        versions.append(("suave", sharp1))
        
        # Versão 2: CLAHE médio + binarização Otsu
        clahe2 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        contrast2 = clahe2.apply(denoised)
        _, binary2 = cv2.threshold(contrast2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        sharp2 = cv2.filter2D(binary2, -1, np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]]))
        versions.append(("medio", sharp2))
        
        # Versão 3: CLAHE forte + binarização adaptativa
        clahe3 = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        contrast3 = clahe3.apply(denoised)
        binary3 = cv2.adaptiveThreshold(contrast3, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)
        sharp3 = cv2.filter2D(binary3, -1, np.array([[0, -1, 0], [-1, 6,-1], [0, -1, 0]]))
        versions.append(("forte", sharp3))
        
        # Versão 4: Sem CLAHE, apenas binarização Otsu
        _, binary4 = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        sharp4 = cv2.filter2D(binary4, -1, np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]]))
        versions.append(("sem_clahe", sharp4))
        
        # Versão 5: Alta resolução (300 DPI)
        high_res = self._upscale_image(denoised, target_dpi=300)
        clahe5 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        contrast5 = clahe5.apply(high_res)
        binary5 = cv2.adaptiveThreshold(contrast5, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)
        sharp5 = cv2.filter2D(binary5, -1, np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]]))
        versions.append(("high_res_300", sharp5))
        
        # Versão 6: Muito alta resolução (400 DPI)
        ultra_res = self._upscale_image(denoised, target_dpi=400)
        clahe6 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        contrast6 = clahe6.apply(ultra_res)
        binary6 = cv2.adaptiveThreshold(contrast6, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)
        sharp6 = cv2.filter2D(binary6, -1, np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]]))
        versions.append(("ultra_res_400", sharp6))
        
        # Versão 7: Otimizada com parâmetros ajustados automaticamente
        optimized = self._create_optimized_version(denoised)
        versions.append(("optimized", optimized))
        
        # Versão 8: Específica para documentos italianos antigos
        italian_antique = self._create_italian_antique_version(denoised)
        versions.append(("italian_antique", italian_antique))
        

        

        

        
        # Aplicar redimensionamento e deskew em todas as versões
        min_height = 1000
        processed_versions = []
        
        for name, version in versions:
            # Redimensionar se necessário
            if version.shape[0] < min_height:
                scale = min_height / version.shape[0]
                version = cv2.resize(version, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            
                    # Deskew
        def compute_skew(image):
            edges = cv2.Canny(image, 50, 150)
            lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
            if lines is None:
                return 0
            angles = []
            for line in lines:
                for rho, theta in line:
                    angle = (theta * 180 / np.pi) - 90
                    angles.append(angle)
            if len(angles) == 0:
                return 0
            median_angle = np.median(angles)
            return median_angle
        
        skew_angle = compute_skew(version)
        if abs(skew_angle) > 0.5:
            version = rotate(version, -skew_angle, resize=False, mode='edge', preserve_range=True).astype(np.uint8)
        
        processed_versions.append((name, version))
        
        # Selecionar a melhor versão baseada em métricas de qualidade
        best_version = self._select_best_version(processed_versions)
        logger.info(f"Versão selecionada: {best_version[0]}")
        
        # Salvar a melhor versão
        pre_dir = os.path.join(os.path.dirname(image_path), 'preprocessed')
        os.makedirs(pre_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        preprocessed_path = os.path.join(pre_dir, f"{base_name}_{file_hash}_{best_version[0]}.jpg")
        
        if save_debug or force_reprocess:
            cv2.imwrite(preprocessed_path, best_version[1])
            logger.info(f"Melhor imagem pré-processada salva: {preprocessed_path}")
        
        return preprocessed_path
    
    def _select_best_version(self, versions):
        """
        Seleciona a melhor versão baseada em métricas de qualidade
        """
        import cv2
        import numpy as np
        
        best_score = -1
        best_version = versions[0]  # Fallback para primeira versão
        
        for name, version in versions:
            # Calcular métricas de qualidade
            score = self._calculate_quality_score(version)
            logger.info(f"Versão {name}: score = {score:.2f}")
            
            if score > best_score:
                best_score = score
                best_version = (name, version)
        
        logger.info(f"Melhor versão selecionada: {best_version[0]} (score: {best_score:.2f})")
        return best_version
    
    def _calculate_quality_score(self, image):
        """
        Calcula um score de qualidade para a imagem
        """
        import cv2
        import numpy as np
        
        # 1. Contraste (quanto maior, melhor)
        contrast = np.std(image)
        
        # 2. Nitidez (usando Laplacian)
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        sharpness = np.var(laplacian)
        
        # 3. Proporção de pixels pretos vs brancos (ideal: ~50/50)
        black_pixels = np.sum(image == 0)
        total_pixels = image.size
        black_ratio = black_pixels / total_pixels
        balance_score = 1.0 - abs(black_ratio - 0.5) * 2  # Máximo quando 50/50
        
        # 4. Uniformidade (menor variação local = melhor)
        # Usar blur para detectar variações locais
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        uniformity = 1.0 / (1.0 + np.std(image - blurred))
        
        # Combinar métricas (pesos podem ser ajustados)
        final_score = (
            contrast * 0.3 +
            sharpness * 0.3 +
            balance_score * 0.2 +
            uniformity * 0.2
        )
        
        return final_score
    
    def _upscale_image(self, image, target_dpi=300):
        """
        Faz upscaling da imagem para melhorar a resolução para OCR
        """
        import cv2
        import numpy as np
        
        # Estimar DPI atual (assumir 150 DPI se não conhecido - mais realista)
        current_dpi = 150
        
        # Calcular fator de escala
        scale_factor = target_dpi / current_dpi
        
        # Calcular novas dimensões
        height, width = image.shape[:2]
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        # Limitar tamanho máximo para evitar problemas de memória
        max_dimension = 4000
        if new_width > max_dimension or new_height > max_dimension:
            # Reduzir escala para caber no limite
            scale_factor = min(max_dimension / width, max_dimension / height)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            target_dpi = int(current_dpi * scale_factor)
        
        # Upscaling com interpolação bicúbica (melhor qualidade)
        upscaled = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        
        logger.info(f"Upscaling: {width}x{height} -> {new_width}x{new_height} (DPI: {current_dpi} -> {target_dpi})")
        
        return upscaled
    
    def _create_optimized_version(self, image):
        """
        Cria uma versão otimizada com parâmetros ajustados automaticamente
        """
        import cv2
        import numpy as np
        
        # 1. Upscaling otimizado (testar apenas 3 DPIs mais promissores)
        best_dpi = self._find_best_dpi_fast(image)
        upscaled = self._upscale_image(image, target_dpi=best_dpi)
        
        # 2. CLAHE otimizado (testar apenas 3 valores)
        best_clip_limit = self._find_best_clahe_fast(upscaled)
        clahe = cv2.createCLAHE(clipLimit=best_clip_limit, tileGridSize=(8,8))
        contrast = clahe.apply(upscaled)
        
        # 3. Binarização otimizada (testar apenas 2 métodos)
        best_binary = self._find_best_binarization_fast(contrast)
        
        # 4. Sharpening simples (kernel médio)
        sharpened = cv2.filter2D(best_binary, -1, np.array([[0, -1, 0], [-1, 6, -1], [0, -1, 0]]))
        
        logger.info(f"Versão otimizada: DPI={best_dpi}, CLAHE={best_clip_limit}")
        
        return sharpened
    
    def _find_best_dpi(self, image):
        """
        Encontra o melhor DPI para a imagem
        """
        dpi_options = [200, 250, 300, 350, 400]
        best_score = -1
        best_dpi = 300  # default
        
        for dpi in dpi_options:
            try:
                upscaled = self._upscale_image(image, target_dpi=dpi)
                score = self._calculate_quality_score(upscaled)
                if score > best_score:
                    best_score = score
                    best_dpi = dpi
            except:
                continue
        
        return best_dpi
    
    def _find_best_clahe_params(self, image):
        """
        Encontra os melhores parâmetros CLAHE
        """
        clip_limits = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
        best_score = -1
        best_clip = 2.0  # default
        
        for clip in clip_limits:
            try:
                clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8,8))
                enhanced = clahe.apply(image)
                score = self._calculate_quality_score(enhanced)
                if score > best_score:
                    best_score = score
                    best_clip = clip
            except:
                continue
        
        return best_clip
    
    def _find_best_binarization(self, image):
        """
        Encontra o melhor método de binarização
        """
        # Método 1: Adaptive Gaussian
        try:
            binary1 = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)
            score1 = self._calculate_quality_score(binary1)
        except:
            score1 = -1
        
        # Método 2: Otsu
        try:
            _, binary2 = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            score2 = self._calculate_quality_score(binary2)
        except:
            score2 = -1
        
        # Método 3: Adaptive Mean
        try:
            binary3 = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 31, 15)
            score3 = self._calculate_quality_score(binary3)
        except:
            score3 = -1
        
        # Retornar o melhor
        scores = [(score1, binary1), (score2, binary2), (score3, binary3)]
        best_score, best_binary = max(scores, key=lambda x: x[0] if x[0] > -1 else -1)
        
        return best_binary
    
    def _find_best_sharpening(self, image):
        """
        Encontra o melhor kernel de sharpening
        """
        kernels = [
            np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),  # Laplacian
            np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]),  # Strong
            np.array([[0, -1, 0], [-1, 6, -1], [0, -1, 0]]),  # Medium
            np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]])  # Soft
        ]
        
        best_score = -1
        best_sharp = image  # sem sharpening como fallback
        
        for kernel in kernels:
            try:
                sharpened = cv2.filter2D(image, -1, kernel)
                score = self._calculate_quality_score(sharpened)
                if score > best_score:
                    best_score = score
                    best_sharp = sharpened
            except:
                continue
        
        return best_sharp
    
    def _create_italian_antique_version(self, image):
        """
        Cria uma versão específica para documentos italianos antigos
        """
        import cv2
        import numpy as np
        
        # 1. Upscaling para alta resolução (documentos antigos precisam de mais detalhes)
        upscaled = self._upscale_image(image, target_dpi=350)
        
        # 2. Remoção de manchas de papel antigo (filtro bilateral)
        denoised = cv2.bilateralFilter(upscaled, 9, 75, 75)
        
        # 3. CLAHE específico para tinta desbotada (clip limit mais alto)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # 4. Correção de cor para papel amarelado (remover tons amarelos)
        # Converter para HSV e ajustar saturação
        hsv = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        hsv = cv2.cvtColor(hsv, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], 0.7)  # Reduzir saturação
        hsv[:, :, 2] = cv2.multiply(hsv[:, :, 2], 1.2)  # Aumentar brilho
        corrected = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        corrected = cv2.cvtColor(corrected, cv2.COLOR_BGR2GRAY)
        
        # 5. Binarização adaptativa específica para texto antigo
        binary = cv2.adaptiveThreshold(corrected, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 10)
        
        # 6. Remoção de ruído específico de documentos antigos
        # Kernel para remover pontos pequenos
        kernel = np.ones((2,2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # 7. Sharpening específico para caligrafia antiga (kernel mais suave)
        sharpened = cv2.filter2D(cleaned, -1, np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]]))
        
        # 8. Correção final de contraste (mais agressiva)
        final = cv2.convertScaleAbs(sharpened, alpha=1.2, beta=10)
        
        # 9. Ajuste fino adicional para documentos italianos
        # Aplicar um leve blur gaussiano para suavizar ruído residual
        final = cv2.GaussianBlur(final, (3, 3), 0.5)
        
        # 10. Correção final de brilho
        final = cv2.convertScaleAbs(final, alpha=1.0, beta=15)
        
        logger.info("Versão italiana antiga criada com sucesso")
        
        return final
    
    def _find_best_dpi_fast(self, image):
        """
        Encontra o melhor DPI rapidamente (apenas 3 opções)
        """
        dpi_options = [250, 300, 350]  # DPIs mais promissores
        best_score = -1
        best_dpi = 300  # default
        
        for dpi in dpi_options:
            try:
                upscaled = self._upscale_image(image, target_dpi=dpi)
                score = self._calculate_quality_score(upscaled)
                if score > best_score:
                    best_score = score
                    best_dpi = dpi
            except:
                continue
        
        return best_dpi
    
    def _find_best_clahe_fast(self, image):
        """
        Encontra os melhores parâmetros CLAHE rapidamente
        """
        clip_limits = [2.0, 2.5, 3.0]  # Valores mais promissores
        best_score = -1
        best_clip = 2.0  # default
        
        for clip in clip_limits:
            try:
                clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8,8))
                enhanced = clahe.apply(image)
                score = self._calculate_quality_score(enhanced)
                if score > best_score:
                    best_score = score
                    best_clip = clip
            except:
                continue
        
        return best_clip
    
    def _find_best_binarization_fast(self, image):
        """
        Encontra o melhor método de binarização rapidamente
        """
        # Método 1: Adaptive Gaussian
        try:
            binary1 = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)
            score1 = self._calculate_quality_score(binary1)
        except:
            score1 = -1
        
        # Método 2: Otsu
        try:
            _, binary2 = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            score2 = self._calculate_quality_score(binary2)
        except:
            score2 = -1
        
        # Retornar o melhor
        if score1 > score2:
            return binary1
        else:
            return binary2
    

    

    
    def extract_text(self, image_path: str, force_reprocess: bool = True) -> Dict[str, Any]:
        """
        Extrai texto da imagem usando OCR
        """
        try:
            logger.info(f"Iniciando processamento OCR para: {image_path}")
            
            # Pré-processar imagem e salvar (sempre forçar reprocessamento)
            preprocessed_path = self.preprocess_image(image_path, save_debug=True, force_reprocess=force_reprocess)
            processed_image = cv2.imread(preprocessed_path)
            
            if processed_image is None:
                raise ValueError(f"Não foi possível ler a imagem pré-processada: {preprocessed_path}")
            
            # Testar múltiplas configurações Tesseract para encontrar a melhor
            best_confidence = 0
            best_text = ""
            best_data = None
            best_config = ""
            
            # Configurações para testar (específicas para documentos antigos)
            configs = [
                ('--oem 3 --psm 6 -l ita', 'PSM6_OEM3'),
                ('--oem 1 --psm 6 -l ita', 'PSM6_OEM1'),
                ('--oem 3 --psm 4 -l ita', 'PSM4_OEM3'),  # Assumir coluna única
                ('--oem 3 --psm 5 -l ita', 'PSM5_OEM3'),  # Bloco uniforme
                ('--oem 3 --psm 3 -l ita', 'PSM3_OEM3'),  # Página completa
            ]
            
            for config, config_name in configs:
                try:
                    # Extrair texto
                    text = pytesseract.image_to_string(
                        processed_image, 
                        config=config,
                        lang=settings.OCR_LANG
                    )
                    
                    # Obter dados de confiança
                    data = pytesseract.image_to_data(
                        processed_image, 
                        config=config,
                        lang=settings.OCR_LANG,
                        output_type=pytesseract.Output.DICT
                    )
                    
                    # Calcular confiança média
                    confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                    avg_confidence = np.mean(confidences) if confidences else 0.0
                    
                    # Atualizar melhor resultado
                    if avg_confidence > best_confidence:
                        best_confidence = avg_confidence
                        best_text = text
                        best_data = data
                        best_config = config_name
                        
                    logger.info(f"Config {config_name}: {avg_confidence:.2f}%")
                    
                except Exception as e:
                    logger.warning(f"Erro na config {config_name}: {str(e)}")
                    continue
            
            logger.info(f"Melhor configuração: {best_config} ({best_confidence:.2f}%)")
            
            # Usar o melhor resultado
            text = best_text
            data = best_data
            avg_confidence = best_confidence
            

            
            # Limpar texto
            cleaned_text = self.clean_text(text)
            
            logger.info(f"OCR concluído com confiança: {avg_confidence:.2f}")
            
            # Não limpar cache durante o processamento para evitar interferência
            # O cache será limpo pelo endpoint após salvar no banco
            
            return {
                'text': cleaned_text,
                'confidence': avg_confidence,
                'raw_text': text,
                'word_confidences': data['conf'],
                'preprocessed_path': preprocessed_path
            }
            
        except Exception as e:
            logger.error(f"Erro no OCR: {str(e)}")
            raise
    
    def clean_text(self, text: str) -> str:
        """
        Limpa e normaliza o texto extraído
        """
        # Remover quebras de linha extras
        text = ' '.join(text.split())
        
        # Remover caracteres especiais problemáticos
        text = text.replace('|', 'I')  # Comum em OCR
        text = text.replace('0', 'O')  # Em alguns contextos
        
        # Normalizar espaços
        text = ' '.join(text.split())
        
        return text.strip()
    
    def extract_text_with_confidence(self, image_path: str) -> Tuple[str, float]:
        """
        Versão simplificada que retorna apenas texto e confiança
        """
        result = self.extract_text(image_path)
        return result['text'], result['confidence']
    
    def batch_process(self, image_paths: list) -> list:
        """
        Processa múltiplas imagens em lote
        """
        results = []
        for image_path in image_paths:
            try:
                result = self.extract_text(image_path)
                results.append({
                    'image_path': image_path,
                    'success': True,
                    **result
                })
            except Exception as e:
                results.append({
                    'image_path': image_path,
                    'success': False,
                    'error': str(e)
                })
        
        return results
    
    def clear_cache(self, keep_recent: bool = True):
        """
        Limpa o cache de imagens pré-processadas
        keep_recent: se True, mantém os arquivos mais recentes (últimos 5 minutos)
        """
        pre_dir = os.path.join(settings.UPLOAD_DIR, 'preprocessed')
        if not os.path.exists(pre_dir):
            return
        
        current_time = time.time()
        files_removed = 0
        
        for file in os.listdir(pre_dir):
            file_path = os.path.join(pre_dir, file)
            try:
                if os.path.isfile(file_path):
                    # Se keep_recent=True, manter arquivos criados nos últimos 5 minutos
                    if keep_recent:
                        file_age = current_time - os.path.getctime(file_path)
                        if file_age < 300:  # 5 minutos
                            continue
                    
                    os.remove(file_path)
                    files_removed += 1
                    logger.debug(f"Cache removido: {file_path}")
            except Exception as e:
                logger.warning(f"Erro ao remover arquivo de cache: {file_path} - {str(e)}")
        
        if files_removed > 0:
            logger.info(f"Cache limpo: {files_removed} arquivos removidos")
        else:
            logger.debug("Cache já estava limpo")

# Instância global do serviço
ocr_service = OCRService() 