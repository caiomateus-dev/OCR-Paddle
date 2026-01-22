import cv2
import os
import io
import tempfile
import fitz

from pathlib import Path
from PIL import Image

class ImageManipulatorService:
    def __init__(self):
        self.max_size_mb = 10 
        self.max_size_px = 1500 

    def convert_to_png(self, input_path: str, password: str = None) -> Image.Image | list[Image.Image]:
        try:
            file_extension = os.path.splitext(input_path)[1].lower()
            
            if file_extension == '.pdf':
                return self._convert_pdf_to_png(input_path, password=password)
            elif file_extension in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
                return self._convert_image_to_png(input_path)
            else:
                raise ValueError(f"Formato de arquivo não suportado: {file_extension}")
                
        except Exception as e:
            raise

    def _convert_pdf_to_png(self, pdf_path: str, password: str = None) -> list[Image.Image]:
        pdf_document = None
        try:
            # Abrir PDF
            pdf_document = fitz.open(pdf_path)
            
            # Autenticar com senha se fornecida e necessária
            if pdf_document.needs_pass:
                if password:
                    if not pdf_document.authenticate(password):
                        raise ValueError(f"Senha incorreta para PDF: {pdf_path}")
                else:
                    raise ValueError(f"PDF requer senha mas nenhuma foi fornecida: {pdf_path}")
            
            images = []
            
            for page_num in range(len(pdf_document)):
                page = pdf_document.load_page(page_num)
                
                mat = fitz.Matrix(3.0, 3.0)  
                pix = page.get_pixmap(matrix=mat, alpha=False)
                
                img_data = pix.tobytes("png")
                img = Image.open(io.BytesIO(img_data))
                images.append(img)
                
            
            return images
            
        except Exception as e:
            raise
        finally:
            # SEMPRE fechar o PDF, mesmo em caso de exceção
            if pdf_document is not None:
                try:
                    pdf_document.close()
                except Exception as close_error:
                    raise
    def _convert_image_to_png(self, image_path: str) -> Image.Image:
        try:
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f"OpenCV não conseguiu ler a imagem: {image_path}")
            
            if len(img.shape) == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            elif len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            img_buffer = io.BytesIO()
            pil_img = Image.fromarray(img)
            pil_img.save(img_buffer, format='PNG', optimize=False)
            img_buffer.seek(0)
            
            png_img = Image.open(img_buffer)
            return png_img
                
        except Exception as e:
            try:
                with Image.open(image_path) as img:
                    if img.mode in ('CMYK', 'RGBA'):
                        if img.mode == 'CMYK':
                            img = img.convert('RGB')
                        elif img.mode == 'RGBA':
                            pass
                    
                    img_buffer = io.BytesIO()
                    img.save(img_buffer, format='PNG', optimize=False)
                    img_buffer.seek(0)
                    
                    png_img = Image.open(img_buffer)
                    return png_img
            except Exception as pil_error:
                raise

    def _needs_resizing(self, image_path):
        if not os.path.exists(image_path):
            return False
            
        file_size_mb = os.path.getsize(image_path) / (1024 * 1024)
        
        if file_size_mb <= self.max_size_mb:
            return False
            
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return False
            
        (height, width) = img.shape[:2]
        if width <= self.max_size_px and height <= self.max_size_px:
            return False
            
        return True

    def resize_image(self, image_path):
        if not self._needs_resizing(image_path):
            return image_path
            
        try:
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                return None
                
            (height, width) = img.shape[:2]
            scaling_factor = min(self.max_size_px / float(width), self.max_size_px / float(height))
            new_size = (int(width * scaling_factor), int(height * scaling_factor))
            
            temp_file = tempfile.NamedTemporaryFile(
                suffix=Path(image_path).suffix,
                delete=False,
                prefix="resized_"
            )
            temp_path = temp_file.name
            temp_file.close()
            
            resized_img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
            cv2.imwrite(temp_path, resized_img)
            
            return temp_path
            
        except Exception as e:
            return None

    def crop_image(self, image_path, crop_area):
        if not os.path.exists(image_path):
            return None
            
        if not (isinstance(crop_area, tuple) and len(crop_area) == 4):
            return None
            
        try:
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                return None
                
            cropped_img = img[crop_area[1]:crop_area[3], crop_area[0]:crop_area[2]]
            
            temp_file = tempfile.NamedTemporaryFile(
                suffix=Path(image_path).suffix,
                delete=False,
                prefix="cropped_"
            )
            temp_path = temp_file.name
            temp_file.close()
            
            cv2.imwrite(temp_path, cropped_img)
            
            return temp_path
            
        except Exception as e:
            return None

    def rotate_image(self, image_path, angle):
        """
        Rotaciona uma imagem
        """
        if not os.path.exists(image_path):
            return None
            
        try:
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                return None
            
            # Obter dimensões da imagem
            height, width = img.shape[:2]
            
            # Calcular o centro da imagem
            center = (width // 2, height // 2)
            
            # Obter matriz de rotação
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            # Calcular novas dimensões
            abs_cos = abs(rotation_matrix[0, 0])
            abs_sin = abs(rotation_matrix[0, 1])
            
            bound_w = int(height * abs_sin + width * abs_cos)
            bound_h = int(height * abs_cos + width * abs_sin)
            
            # Ajustar matriz de rotação para manter a imagem completa
            rotation_matrix[0, 2] += bound_w / 2 - center[0]
            rotation_matrix[1, 2] += bound_h / 2 - center[1]
            
            # Aplicar rotação
            rotated_img = cv2.warpAffine(img, rotation_matrix, (bound_w, bound_h))
            
            temp_file = tempfile.NamedTemporaryFile(
                suffix=Path(image_path).suffix,
                delete=False,
                prefix="rotated_"
            )
            temp_path = temp_file.name
            temp_file.close()
            
            cv2.imwrite(temp_path, rotated_img)
            
            return temp_path
            
        except Exception as e:
            return None

    def mask_area(self, image_path, mask_area, color=(0, 0, 0)):
        if not os.path.exists(image_path):
            return None
            
        if not (isinstance(mask_area, tuple) and len(mask_area) == 4):
            return None
            
        try:
            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                return None
                
            masked_img = img.copy()
            cv2.rectangle(masked_img, (mask_area[0], mask_area[1]), (mask_area[2], mask_area[3]), color, thickness=cv2.FILLED)
            
            temp_file = tempfile.NamedTemporaryFile(
                suffix=Path(image_path).suffix,
                delete=False,
                prefix="masked_"
            )
            temp_path = temp_file.name
            temp_file.close()
            
            cv2.imwrite(temp_path, masked_img)
            
            return temp_path
            
        except Exception as e:
            return None

    def save_numpy_array_as_image(self, np_array, prefix="crop_"):
        """
        Converte um numpy array em imagem e salva em arquivo temporário.
        
        Args:
            np_array: numpy array da imagem
            prefix: prefixo para o nome do arquivo temporário
            
        Returns:
            Caminho do arquivo temporário criado ou None em caso de erro
        """
        if np_array is None:
            return None
            
        try:
            import numpy as np
            
            # Verificar se é um numpy array
            if not isinstance(np_array, np.ndarray):
                return None
            
            # Converter para PIL Image
            if len(np_array.shape) == 2:
                # Imagem em escala de cinza
                pil_img = Image.fromarray(np_array, mode='L')
            elif len(np_array.shape) == 3:
                # Imagem colorida
                if np_array.shape[2] == 3:
                    # RGB
                    pil_img = Image.fromarray(np_array, mode='RGB')
                elif np_array.shape[2] == 4:
                    # RGBA
                    pil_img = Image.fromarray(np_array, mode='RGBA')
                else:
                    return None
            else:
                return None
            
            # Criar arquivo temporário
            temp_file = tempfile.NamedTemporaryFile(
                suffix='.png',
                delete=False,
                prefix=prefix
            )
            temp_path = temp_file.name
            temp_file.close()
            
            # Salvar imagem
            pil_img.save(temp_path, format='PNG')
            
            return temp_path
            
        except Exception as e:
            return None