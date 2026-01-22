import os
import sys

os.environ['MKL_NUM_THREADS'] = '1'
os.environ['MKL_DYNAMIC'] = 'FALSE'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMBA_NUM_THREADS'] = '1'

os.environ['OMP_WAIT_POLICY'] = 'PASSIVE'

import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

torch.backends.mkldnn.enabled = False
torch.backends.mkldnn.allow_tf32 = False

from ultralytics import YOLO

if hasattr(torch.utils.data, 'get_worker_info'):
    pass

from utils.image_manipulator_service import ImageManipulatorService
from utils.temp_file_manager import TempFileManager

consumption_path = os.path.join(os.path.dirname(__file__), "models", "consumption.pt")
customer_data_detector_path = os.path.join(os.path.dirname(__file__), "models", "customer_data_detector.pt")

_yolo_consumption_model_cache = None
_yolo_customer_data_model_cache = None

def get_yolo_consumption_model():
    """Retorna uma instância singleton do modelo YOLO para consumo"""
    global _yolo_consumption_model_cache
    if _yolo_consumption_model_cache is None:
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)
        
        _yolo_consumption_model_cache = YOLO(consumption_path)
        
        if hasattr(_yolo_consumption_model_cache, 'model'):
            _yolo_consumption_model_cache.model.eval()
            for module in _yolo_consumption_model_cache.model.modules():
                if hasattr(module, 'num_threads'):
                    module.num_threads = 1
        
    return _yolo_consumption_model_cache

def get_yolo_customer_data_model():
    """Retorna uma instância singleton do modelo YOLO para dados do cliente"""
    global _yolo_customer_data_model_cache
    if _yolo_customer_data_model_cache is None:
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        import warnings
        warnings.filterwarnings("ignore", category=UserWarning)
        
        _yolo_customer_data_model_cache = YOLO(customer_data_detector_path)
        
        if hasattr(_yolo_customer_data_model_cache, 'model'):
            _yolo_customer_data_model_cache.model.eval()
            for module in _yolo_customer_data_model_cache.model.modules():
                if hasattr(module, 'num_threads'):
                    module.num_threads = 1
        
    return _yolo_customer_data_model_cache

class ObjectDetection:

    def __init__(self):
        self.consumption_model = get_yolo_consumption_model()  # Usar singleton
        self.customer_data_model = get_yolo_customer_data_model()  # Usar singleton
        self.image_manipulator = ImageManipulatorService()
        self.temp_file_manager = TempFileManager()

    def detect_consumption_objects(self, image_path):
        """Detecta objetos de consumo na imagem"""
        try:
            consumption_results = self.consumption_model(
                image_path, 
                conf=0.25, 
                iou=0.45,
                verbose=False
            )
            return consumption_results
        except RuntimeError as e:
            error_msg = str(e)
            if "could not create a primitive" in error_msg:
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                else:
                    raise RuntimeError(f"YOLO inference failed: PyTorch threading conflict in multiprocessing environment. Original error: {error_msg}")

    def detect_customer_data_objects(self, image_path):
        """Detecta objetos de dados do cliente na imagem"""
        try:
            customer_data_results = self.customer_data_model(
                image_path, 
                conf=0.25, 
                iou=0.45,
                verbose=False
            )
            return customer_data_results
        except RuntimeError as e:
            error_msg = str(e)
            if "could not create a primitive" in error_msg:
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                else:
                    raise RuntimeError(f"YOLO inference failed: PyTorch threading conflict in multiprocessing environment. Original error: {error_msg}")

    def detect_and_crop_consumption(self, image_path):
        """
        Detecta objetos de consumo na imagem e retorna path do recorte
        
        Args:
            image_path (str): Caminho da imagem original
            
        Returns:
            str: Path do recorte de consumo ou None se não encontrar objetos
        """
        consumption_results = self.detect_consumption_objects(image_path)
        
        consumption_crop_path = None

        if consumption_results and len(consumption_results) > 0:
            for result in consumption_results:
                if result.boxes is not None and len(result.boxes) > 0:
                    if len(result.boxes.xyxy) > 0:
                        confs = result.boxes.conf.cpu().numpy()
                        best_idx = int(confs.argmax())

                        bbox = result.boxes.xyxy[best_idx].tolist()
                        
                        if len(bbox) == 4 and all(isinstance(coord, (int, float)) for coord in bbox):
                            bbox_tuple = tuple(int(coord) for coord in bbox)
                            consumption_crop_path = self.image_manipulator.crop_image(image_path, bbox_tuple)
                            if consumption_crop_path:
                                self.temp_file_manager.add_temp_file(consumption_crop_path)
                            break
                        else:
                            raise ValueError(f"Bbox inválido: {bbox} - deve ter 4 coordenadas (x1, y1, x2, y2)")
                    else:
                        raise ValueError("Nenhum objeto detectado na imagem")
        return consumption_crop_path

    def detect_and_crop_customer_data(self, image_path):
        """
        Detecta objetos de dados do cliente na imagem e retorna path do recorte
        
        Args:
            image_path (str): Caminho da imagem original
            
        Returns:
            str: Path do recorte de dados do cliente ou None se não encontrar objetos
        """
        customer_data_results = self.detect_customer_data_objects(image_path)
        
        customer_data_crop_path = None

        if customer_data_results and len(customer_data_results) > 0:
            for result in customer_data_results:
                if result.boxes is not None and len(result.boxes) > 0:
                    if len(result.boxes.xyxy) > 0:
                        confs = result.boxes.conf.cpu().numpy()
                        best_idx = int(confs.argmax())

                        bbox = result.boxes.xyxy[best_idx].tolist()
                        
                        if len(bbox) == 4 and all(isinstance(coord, (int, float)) for coord in bbox):
                            bbox_tuple = tuple(int(coord) for coord in bbox)
                            customer_data_crop_path = self.image_manipulator.crop_image(image_path, bbox_tuple)
                            if customer_data_crop_path:
                                self.temp_file_manager.add_temp_file(customer_data_crop_path)
                            break
                        else:
                            raise ValueError(f"Bbox inválido: {bbox} - deve ter 4 coordenadas (x1, y1, x2, y2)")
                    else:
                        raise ValueError("Nenhum objeto detectado na imagem")
        return customer_data_crop_path

    def cleanup_temp_files(self):
        """
        Remove todos os arquivos temporários criados durante as detecções
        """
        self.temp_file_manager.cleanup_temp_files()

    def get_temp_files_info(self):
        """
        Retorna informações sobre os arquivos temporários criados
        """
        return self.temp_file_manager.get_temp_files_info()

    def __del__(self):
        """
        Destructor para garantir limpeza dos arquivos temporários
        """
        try:
            self.temp_file_manager.cleanup_temp_files()
        except Exception:
            pass  # Ignorar erros durante a limpeza no destructor
        