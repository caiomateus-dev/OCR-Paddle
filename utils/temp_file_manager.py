import os

class TempFileManager:
    """
    Gerenciador de arquivos temporários para rastrear e limpar arquivos criados durante o processamento
    """
    
    def __init__(self):
        self.temp_files = []
    
    def add_temp_file(self, file_path: str):
        """
        Adiciona um arquivo temporário à lista de rastreamento
        
        Args:
            file_path (str): Caminho do arquivo temporário
        """
        if file_path and file_path not in self.temp_files:
            self.temp_files.append(file_path)

    
    def remove_temp_file(self, file_path: str):
        """
        Remove um arquivo temporário da lista de rastreamento
        
        Args:
            file_path (str): Caminho do arquivo temporário
        """
        if file_path in self.temp_files:
            self.temp_files.remove(file_path)
    
    def cleanup_temp_files(self):
        """
        Remove todos os arquivos temporários criados durante o processamento
        """
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                raise
        
        self.temp_files.clear()
    
    def get_temp_files_info(self):
        """
        Retorna informações sobre os arquivos temporários criados
        
        Returns:
            List[str]: Lista de caminhos dos arquivos temporários
        """
        return self.temp_files.copy()
    
    def get_temp_files_count(self):
        """
        Retorna o número de arquivos temporários sendo rastreados
        
        Returns:
            int: Número de arquivos temporários
        """
        return len(self.temp_files)
    
    def __del__(self):
        """
        Destructor para garantir limpeza dos arquivos temporários
        """
        try:
            self.cleanup_temp_files()
        except Exception:
            pass  # Ignorar erros durante a limpeza no destructor