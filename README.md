<h1 align="center">Moji Mosaic 2.0</h1>
<p align="center"><i>AI-Powered Text-to-Artistic-Image Generator</i></p>
<div align="center">

![Static Badge](https://img.shields.io/badge/python-3.11+-blue)
![Static Badge](https://img.shields.io/badge/AI-BERT%20%2B%20Diffusion-green)
![Static Badge](https://img.shields.io/badge/vLLM-Accelerated-orange)

</div>

**Moji Mosaic 2.0** is a revolutionary AI-powered text-to-artistic-image generator that transforms text into stunning artistic compositions using state-of-the-art machine learning models. This modernized version combines traditional mosaic generation with cutting-edge BERT keyword extraction and diffusion model artistic generation.

## 🚀 New AI-Powered Workflow

### Core Pipeline
1. **📝 Text Input**: Accept any text content
2. **🎨 Mosaic Generation**: Create colorful pixel mosaics with modern algorithms
3. **🧠 BERT Keyword Extraction**: Extract 5 key themes using state-of-the-art NLP models via vLLM
4. **🎭 Diffusion Art Generation**: Transform mosaics into artistic masterpieces using diffusion models
5. **✨ Final Output**: Stunning artistic images that blend your text's essence with AI creativity

### Key Features
- **🤖 AI-Powered**: Uses BERT for semantic understanding and diffusion models for artistic generation
- **⚡ vLLM Acceleration**: Optimized inference with vLLM for both NLP and image generation
- **🔄 Batch Processing**: Process multiple texts simultaneously with comprehensive progress tracking
- **🌐 Web API**: RESTful API with FastAPI for web integration and background processing
- **💻 Modern CLI**: Beautiful command-line interface with multiple commands and options

## 🛠️ Installation & Setup

### Prerequisites
- **Python 3.11+** (recommended for optimal performance)
- **CUDA-compatible GPU** (optional but recommended for faster AI inference)
- **8GB+ RAM** (16GB+ recommended for large models)

### Quick Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/moji-mosaic.git
cd moji-mosaic
```

2. **Install the package:**
```bash
# Install in development mode with all dependencies
pip install -e .

# Or install from PyPI (when available)
pip install moji-mosaic
```

3. **Install optional dependencies for GPU acceleration:**
```bash
# For CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For additional optimizations
pip install xformers  # Optional: for memory-efficient attention
```

## 🚀 Quick Start

### Command Line Interface

Generate a single artistic image:
```bash
moji-mosaic generate "Your inspiring text here" --output ./my_art --save-intermediates
```

Generate with specific tile shape:
```bash
moji-mosaic generate "Beautiful sunset" --tile-shape circle --keywords 7
```

Generate multiple variations:
```bash
moji-mosaic variations "Beautiful sunset over mountains" --num-variations 5 --output ./variations
```

Batch process multiple texts:
```bash
moji-mosaic batch --texts "Text 1" "Text 2" "Text 3" --output ./batch_results
```

Custom generation parameters:
```bash
moji-mosaic generate "Mystical forest" --mosaic-size 2048 --strength 0.8 --seed 42
```

### Python API

```python
from moji_mosaic import MojiMosaicPipeline
from pathlib import Path

# Initialize the pipeline
pipeline = MojiMosaicPipeline()

# Generate artwork with circle tiles
result = pipeline.generate_complete_artwork(
    text="The quick brown fox jumps over the lazy dog",
    num_keywords=5,
    tile_shape="circle",  # or "square"
    save_intermediates=True,
    output_dir=Path("./output")
)

# Save the result
if result['final_image']:
    result['final_image'].save("my_artwork.png")
    print(f"Keywords: {result['intermediate_results']['keywords']}")
    print(f"Processing time: {result['timing']['total_time']:.2f}s")
```

### Web API Server

Start the FastAPI server:
```bash
# Start the API server
uvicorn moji_mosaic.api:app --host 0.0.0.0 --port 8000

# Or use the development server
python -m moji_mosaic.api
```

Then use the API:
```bash
# Generate artwork via API
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "num_keywords": 5, "seed": 42}'

# Generate variations
curl -X POST "http://localhost:8000/variations" \
  -H "Content-Type: application/json" \
  -d '{"text": "Beautiful sunset", "num_variations": 3}'

# Check job status
curl "http://localhost:8000/status/{job_id}"
```

## ⚙️ Configuration Options

### Mosaic Generation
- `mosaic_size`: Output size (64-2048px, default: 1024)
- `tile_shape`: Tile shape ("square" or "circle", default: "square")
- `preserve_aspect_ratio`: Maintain original text proportions

### Keyword Extraction  
- `num_keywords`: Number of keywords to extract (1-10, default: 5)
- `extraction_method`: "bert", "tfidf", or "hybrid"
- `use_vllm`: Enable vLLM acceleration (default: True)

### Diffusion Generation
- `strength`: Transformation strength (0.0-1.0, default: 0.7)
- `guidance_scale`: Prompt adherence (1.0-200.0, default: 100)
- `num_inference_steps`: Quality vs speed tradeoff (10-200, default: 200)
- `seed`: Random seed for reproducible results
- `tile_size`: Automatic tile size detection (calculated from mosaic)

## 🔧 Advanced Usage

### Custom Pipeline Configuration

```python
from moji_mosaic import MojiMosaicPipeline
from pathlib import Path

# Custom configuration for each component
pipeline = MojiMosaicPipeline(
    mosaic_config={
        'default_size': 2048,
        'min_resolution': 128,
        'default_tile_shape': 'circle',  # Set default tile shape
    },
    keyword_config={
        'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
        'max_keywords': 7,
        'use_vllm': True,
    },
    diffusion_config={
        'model_name': 'stabilityai/stable-diffusion-3.5-medium',  # Updated model
        'device': 'cuda',
        'enable_cpu_offload': True,
        'enable_attention_slicing': True,
    },
    cache_dir=Path('./custom_cache')
)
```

### Batch Processing with Different Styles

```python
texts = [
    "A serene mountain landscape at dawn",
    "Bustling city streets with neon lights", 
    "Ancient forest with mystical creatures"
]

results = pipeline.batch_process(
    texts=texts,
    output_base_dir=Path('./batch_output'),
    save_intermediates=True,
    tile_shape="circle",  # Use circle tiles for all
    num_keywords=6
)

for i, result in enumerate(results):
    if result['metadata']['success']:
        print(f"✅ Text {i+1} processed successfully")
        print(f"   Keywords: {result['intermediate_results']['keywords']}")
        print(f"   Time: {result['timing']['total_time']:.1f}s")
        print(f"   Tile size used: {result.get('tile_size', 'auto')}px")
```

## 🐳 Docker Deployment

Create a `Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install -e .
EXPOSE 8000

CMD ["uvicorn", "moji_mosaic.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t moji-mosaic .
docker run -p 8000:8000 moji-mosaic
```

<div align="center">
<p><strong>Transform your words into art with AI ✨</strong></p>
</div>
