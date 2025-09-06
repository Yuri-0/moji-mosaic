"""FastAPI server for Moji-mosaic web service."""

import base64
import logging
import tempfile
import uuid
from io import BytesIO
from pathlib import Path
from typing import Optional, Dict, Any

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .pipeline import MojiMosaicPipeline


# Pydantic models for API
class GenerationRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000, description="Input text to process")
    mosaic_size: int = Field(default=1024, ge=64, le=2048, description="Size of intermediate mosaic")
    diffusion_strength: float = Field(default=0.7, ge=0.0, le=1.0, description="Diffusion transformation strength")
    num_keywords: int = Field(default=5, ge=1, le=10, description="Number of keywords to extract")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")
    save_intermediates: bool = Field(default=True, description="Save intermediate results")


class VariationsRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000, description="Input text to process")
    num_variations: int = Field(default=3, ge=1, le=10, description="Number of variations to generate")
    mosaic_size: int = Field(default=1024, ge=64, le=2048, description="Size of intermediate mosaic")
    seed: Optional[int] = Field(default=None, description="Base random seed")


class GenerationResponse(BaseModel):
    job_id: str
    status: str
    message: str
    results: Optional[Dict[str, Any]] = None


# Initialize FastAPI app
app = FastAPI(
    title="Moji-mosaic API",
    description="AI-powered text-to-artistic-image generator using BERT and diffusion models",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
pipeline: Optional[MojiMosaicPipeline] = None
job_storage: Dict[str, Dict[str, Any]] = {}
temp_dir = Path(tempfile.gettempdir()) / "moji_mosaic_api"
temp_dir.mkdir(exist_ok=True)


@app.on_event("startup")
async def startup_event():
    """Initialize the pipeline on startup."""
    global pipeline
    try:
        pipeline = MojiMosaicPipeline()
        logging.info("Moji-mosaic pipeline initialized successfully")
    except Exception as e:
        logging.error(f"Failed to initialize pipeline: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    # Clean up temporary files
    import shutil
    try:
        shutil.rmtree(temp_dir)
    except Exception:
        pass


def process_generation_job(job_id: str, request: GenerationRequest) -> None:
    """Background task to process generation job."""
    try:
        job_storage[job_id]["status"] = "processing"
        
        # Create job-specific output directory
        output_dir = temp_dir / job_id
        output_dir.mkdir(exist_ok=True)
        
        # Generate artwork
        result = pipeline.generate_complete_artwork(
            text=request.text,
            mosaic_size=request.mosaic_size,
            diffusion_strength=request.diffusion_strength,
            num_keywords=request.num_keywords,
            seed=request.seed,
            save_intermediates=request.save_intermediates,
            output_dir=output_dir,
        )
        
        # Convert images to base64 for API response
        if result.get('final_image'):
            buffer = BytesIO()
            result['final_image'].save(buffer, format='PNG')
            result['final_image_b64'] = base64.b64encode(buffer.getvalue()).decode()
            del result['final_image']  # Remove PIL object
        
        if result.get('intermediate_results', {}).get('mosaic_image'):
            buffer = BytesIO()
            result['intermediate_results']['mosaic_image'].save(buffer, format='PNG')
            result['intermediate_results']['mosaic_image_b64'] = base64.b64encode(buffer.getvalue()).decode()
            del result['intermediate_results']['mosaic_image']  # Remove PIL object
        
        job_storage[job_id]["results"] = result
        job_storage[job_id]["status"] = "completed" if result.get('metadata', {}).get('success') else "failed"
        
    except Exception as e:
        logging.error(f"Job {job_id} failed: {e}")
        job_storage[job_id]["status"] = "failed"
        job_storage[job_id]["error"] = str(e)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Moji-mosaic API",
        "version": "2.0.0",
        "description": "AI-powered text-to-artistic-image generator",
        "endpoints": {
            "generate": "/generate",
            "variations": "/variations",
            "status": "/status/{job_id}",
            "download": "/download/{job_id}/{file_type}",
            "info": "/info",
            "docs": "/docs",
        }
    }


@app.post("/generate", response_model=GenerationResponse)
async def generate_artwork(request: GenerationRequest, background_tasks: BackgroundTasks):
    """Generate artistic image from text."""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    # Create job
    job_id = str(uuid.uuid4())
    job_storage[job_id] = {
        "status": "queued",
        "request": request.dict(),
        "created_at": str(Path().cwd()),  # Placeholder for timestamp
    }
    
    # Start background processing
    background_tasks.add_task(process_generation_job, job_id, request)
    
    return GenerationResponse(
        job_id=job_id,
        status="queued",
        message="Generation job started. Check status with /status/{job_id}"
    )


@app.post("/variations", response_model=GenerationResponse)
async def generate_variations(request: VariationsRequest, background_tasks: BackgroundTasks):
    """Generate multiple artistic variations from text."""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    # Create job
    job_id = str(uuid.uuid4())
    job_storage[job_id] = {
        "status": "queued",
        "request": request.dict(),
        "type": "variations",
    }
    
    # Start background processing (simplified for variations)
    async def process_variations():
        try:
            job_storage[job_id]["status"] = "processing"
            
            output_dir = temp_dir / job_id
            output_dir.mkdir(exist_ok=True)
            
            results = pipeline.generate_variations(
                text=request.text,
                num_variations=request.num_variations,
                mosaic_size=request.mosaic_size,
                seed=request.seed,
                save_intermediates=True,
                output_dir=output_dir,
            )
            
            # Process results (convert images to base64)
            for result in results:
                if result.get('final_image'):
                    buffer = BytesIO()
                    result['final_image'].save(buffer, format='PNG')
                    result['final_image_b64'] = base64.b64encode(buffer.getvalue()).decode()
                    del result['final_image']
            
            job_storage[job_id]["results"] = results
            job_storage[job_id]["status"] = "completed"
            
        except Exception as e:
            job_storage[job_id]["status"] = "failed"
            job_storage[job_id]["error"] = str(e)
    
    background_tasks.add_task(process_variations)
    
    return GenerationResponse(
        job_id=job_id,
        status="queued",
        message="Variations job started. Check status with /status/{job_id}"
    )


@app.get("/status/{job_id}", response_model=GenerationResponse)
async def get_job_status(job_id: str):
    """Get the status of a generation job."""
    if job_id not in job_storage:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = job_storage[job_id]
    return GenerationResponse(
        job_id=job_id,
        status=job["status"],
        message=f"Job is {job['status']}",
        results=job.get("results")
    )


@app.get("/download/{job_id}/{file_type}")
async def download_file(job_id: str, file_type: str):
    """Download generated files."""
    if job_id not in job_storage:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_dir = temp_dir / job_id
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="Job files not found")
    
    file_map = {
        "mosaic": "01_mosaic.png",
        "final": "02_final_artistic.png",
        "metadata": "metadata.json",
    }
    
    if file_type not in file_map:
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    file_path = job_dir / file_map[file_type]
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=file_path,
        filename=f"{job_id}_{file_map[file_type]}",
        media_type="application/octet-stream"
    )


@app.get("/info")
async def get_pipeline_info():
    """Get pipeline information and system status."""
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    info = pipeline.get_pipeline_info()
    info["api_version"] = "2.0.0"
    info["active_jobs"] = len([j for j in job_storage.values() if j["status"] in ["queued", "processing"]])
    info["total_jobs"] = len(job_storage)
    
    return info


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its associated files."""
    if job_id not in job_storage:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Remove job data
    del job_storage[job_id]
    
    # Remove job files
    job_dir = temp_dir / job_id
    if job_dir.exists():
        import shutil
        shutil.rmtree(job_dir)
    
    return {"message": f"Job {job_id} deleted successfully"}


@app.get("/jobs")
async def list_jobs():
    """List all jobs with their status."""
    return {
        "jobs": [
            {
                "job_id": job_id,
                "status": job["status"],
                "type": job.get("type", "generation"),
            }
            for job_id, job in job_storage.items()
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
