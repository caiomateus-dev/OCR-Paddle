## API (FastAPI)

Suba o servidor:

Use o `uvicorn` do seu `.venv` (senão ele roda no Python global e não encontra `paddleocr`):

- `source .venv/bin/activate`
- `python -m uvicorn main:app --host 0.0.0.0 --port 8102 --reload`

Ou direto:

- `.venv/bin/uvicorn main:app --host 0.0.0.0 --port 8102 --reload`

Endpoint:

- `POST /parse/image` (multipart/form-data)
  - `file`: imagem (png/jpg/...)
  - `concessionaria` (opcional): ex `enel`, `equatorial`, `cpfl`, `cemig`
  - `uf` (opcional): ex `rj`, `sp`, `mg`
  - `dump_extraction` (opcional): `true|false` para salvar `artifacts/ocr_dumps/*.txt` e `*.json` com a extração

A resposta retorna apenas os itens do OCR (texto + bbox): `{"items":[{"text":"...","bbox":[[x,y],[x,y],[x,y],[x,y]]}, ...]}`.
