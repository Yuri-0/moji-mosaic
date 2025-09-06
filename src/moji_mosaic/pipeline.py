"""Main pipeline orchestrating the complete Moji-mosaic workflow."""

import logging
from pathlib import Path
from typing import List, Optional, Dict, Any, Literal
import time

from .core.mosaic_generator import MosaicGenerator
from .core.keyword_extractor import KeywordExtractor
from .core.diffusion_generator import DiffusionGenerator


class MojiMosaicPipeline:
    """Complete pipeline for AI-powered text-to-artistic-image generation."""
    
    def __init__(
        self,
        mosaic_config: Optional[Dict[str, Any]] = None,
        keyword_config: Optional[Dict[str, Any]] = None,
        diffusion_config: Optional[Dict[str, Any]] = None,
        cache_dir: Optional[Path] = None,
    ):
        """Initialize the complete Moji-mosaic pipeline.
        
        Args:
            mosaic_config: Configuration for mosaic generator
            keyword_config: Configuration for keyword extractor
            diffusion_config: Configuration for diffusion generator
            cache_dir: Base cache directory for all components
        """
        self.cache_dir = cache_dir or Path.home() / ".moji_mosaic"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components with configurations
        self.mosaic_generator = MosaicGenerator(
            cache_dir=self.cache_dir / "mosaic",
            **(mosaic_config or {})
        )
        
        self.keyword_extractor = KeywordExtractor(
            cache_dir=self.cache_dir / "keywords",
            **(keyword_config or {})
        )
        
        self.diffusion_generator = DiffusionGenerator(
            cache_dir=self.cache_dir / "diffusion",
            **(diffusion_config or {})
        )
        
        # Setup logging
        self._setup_logging()
        
        logging.info("Moji-mosaic pipeline initialized successfully")
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        log_dir = self.cache_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / "moji_mosaic.log"),
                logging.StreamHandler()
            ]
        )
    
    def generate_complete_artwork(
        self,
        text: str,
        mosaic_size: int = 1024,
        diffusion_strength: float = 0.7,
        num_keywords: int = 5,
        save_intermediates: bool = False,
        output_dir: Optional[Path] = None,
        seed: Optional[int] = None,
        tile_shape: Optional[Literal["square", "circle"]] = None,
    ) -> Dict[str, Any]:
        """Generate complete artistic image from text through the full pipeline.
        
        Args:
            text: Input text to process
            mosaic_size: Size of the intermediate mosaic image
            diffusion_strength: Strength of diffusion transformation
            num_keywords: Number of keywords to extract
            save_intermediates: Whether to save intermediate results
            output_dir: Directory to save outputs
            seed: Random seed for reproducibility
            tile_shape: Shape of individual tiles ('square' or 'circle')
            
        Returns:
            Dictionary containing all results and metadata
        """
        start_time = time.time()
        results = {
            'input_text': text,
            'config': {
                'mosaic_size': mosaic_size,
                'diffusion_strength': diffusion_strength,
                'num_keywords': num_keywords,
                'seed': seed,
                'tile_shape': tile_shape,
            },
            'timing': {},
            'intermediate_results': {},
        }
        
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Step 1: Generate mosaic image
            logging.info("Step 1: Generating mosaic image...")
            step_start = time.time()

            mosaic_image, actual_tile_size = self.mosaic_generator.generate_mosaic(
                text, output_size=mosaic_size, tile_shape=tile_shape
            )

            results['timing']['mosaic_generation'] = time.time() - step_start
            results['intermediate_results']['mosaic_image'] = mosaic_image

            if save_intermediates and output_dir:
                mosaic_path = output_dir / "01_mosaic.png"
                mosaic_image.save(mosaic_path)
                logging.info(f"Saved mosaic image to {mosaic_path}")

            # Step 2: Extract keywords using BERT
            logging.info("Step 2: Extracting keywords with BERT...")
            step_start = time.time()

            # Update keyword extractor max_keywords
            self.keyword_extractor.max_keywords = num_keywords
            keywords = self.keyword_extractor.extract_keywords(text, method="hybrid")

            results['timing']['keyword_extraction'] = time.time() - step_start
            results['intermediate_results']['keywords'] = keywords

            logging.info(f"Extracted keywords: {keywords}")

            # Step 3: Generate artistic image with diffusion model
            logging.info("Step 3: Generating artistic image with diffusion model...")
            step_start = time.time()

            artistic_image = self.diffusion_generator.generate_artistic_image(
                mosaic_image=mosaic_image,
                keywords=keywords,
                strength=diffusion_strength,
                seed=seed,
                tile_size=actual_tile_size,
            )

            results['timing']['diffusion_generation'] = time.time() - step_start
            results['final_image'] = artistic_image

            if artistic_image and save_intermediates and output_dir:
                final_path = output_dir / "02_final_artistic.png"
                artistic_image.save(final_path)
                logging.info(f"Saved final artistic image to {final_path}")

            # Calculate total time
            results['timing']['total_time'] = time.time() - start_time

            # Add metadata
            results['metadata'] = {
                'mosaic_stats': self.mosaic_generator.get_color_statistics(text),
                'keyword_analysis': self.keyword_extractor.analyze_text_themes(text),
                'diffusion_info': self.diffusion_generator.get_model_info(),
                'success': artistic_image is not None,
            }

            logging.info(f"Pipeline completed successfully in {results['timing']['total_time']:.2f}s")
            
        except Exception as e:
            logging.error(f"Pipeline failed: {e}")
            results['error'] = str(e)
            results['metadata'] = {'success': False}
        
        return results
    
    def generate_variations(
        self,
        text: str,
        num_variations: int = 3,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Generate multiple variations from the same text using different seeds.
        
        Args:
            text: Input text to process
            num_variations: Number of variations to generate
            **kwargs: Additional arguments for generate_complete_artwork
            
        Returns:
            List of result dictionaries for each variation
        """
        variations = []
        base_seed = kwargs.get('seed', 42)
        
        for i in range(num_variations):
            variation_seed = base_seed + i if base_seed else None
            
            logging.info(f"Generating variation {i+1}/{num_variations} with seed: {variation_seed}")
            
            result = self.generate_complete_artwork(
                text=text,
                seed=variation_seed,
                **kwargs
            )
            
            result['variation_info'] = {
                'variation_number': i + 1,
                'seed_used': variation_seed,
            }
            
            variations.append(result)
        
        return variations
    
    def batch_process(
        self,
        texts: List[str],
        output_base_dir: Optional[Path] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Process multiple texts in batch.
        
        Args:
            texts: List of input texts to process
            output_base_dir: Base directory for outputs (creates subdirs for each text)
            **kwargs: Additional arguments for generate_complete_artwork
            
        Returns:
            List of result dictionaries for each text
        """
        results = []
        
        for i, text in enumerate(texts):
            logging.info(f"Processing batch item {i+1}/{len(texts)}")
            
            # Create output directory for this text
            if output_base_dir:
                text_hash = str(hash(text))[:8]
                output_dir = Path(output_base_dir) / f"text_{i+1:03d}_{text_hash}"
                kwargs['output_dir'] = output_dir
                kwargs['save_intermediates'] = True
            
            result = self.generate_complete_artwork(text, **kwargs)
            result['batch_info'] = {
                'batch_index': i,
                'total_items': len(texts),
            }
            
            results.append(result)
        
        return results
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the pipeline and its components."""
        return {
            'pipeline_version': '2.0.0',
            'components': {
                'mosaic_generator': {
                    'class': self.mosaic_generator.__class__.__name__,
                    'cache_dir': str(self.mosaic_generator.cache_dir),
                },
                'keyword_extractor': {
                    'class': self.keyword_extractor.__class__.__name__,
                    'model_name': self.keyword_extractor.model_name,
                    'use_vllm': self.keyword_extractor.use_vllm,
                },
                'diffusion_generator': self.diffusion_generator.get_model_info(),
            },
            'cache_dir': str(self.cache_dir),
        }
    
    def cleanup_cache(self, older_than_days: int = 30) -> None:
        """Clean up old cache files.
        
        Args:
            older_than_days: Remove cache files older than this many days
        """
        cutoff_time = time.time() - (older_than_days * 24 * 60 * 60)
        
        for cache_path in self.cache_dir.rglob("*"):
            if cache_path.is_file() and cache_path.stat().st_mtime < cutoff_time:
                try:
                    cache_path.unlink()
                    logging.info(f"Removed old cache file: {cache_path}")
                except OSError:
                    pass  # File might be in use
