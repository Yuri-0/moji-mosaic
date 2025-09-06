"""Modern CLI interface for Moji-mosaic."""

import logging
from pathlib import Path
from typing import Optional

import click

from .pipeline import MojiMosaicPipeline


@click.group()
@click.version_option(version="2.0.0")
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def main(verbose: bool) -> None:
    """Moji-mosaic: AI-powered text-to-artistic-image generator."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)


@main.command()
@click.argument('text', type=str)
@click.option('--output', '-o', type=click.Path(), help='Output directory for generated images')
@click.option('--mosaic-size', default=1024, help='Size of intermediate mosaic image')
@click.option('--strength', default=0.7, type=float, help='Diffusion transformation strength (0.0-1.0)')
@click.option('--keywords', default=5, help='Number of keywords to extract')
@click.option('--seed', type=int, help='Random seed for reproducibility')
@click.option('--save-intermediates', is_flag=True, help='Save intermediate results')
def generate(
    text: str,
    output: Optional[str],
    mosaic_size: int,
    strength: float,
    keywords: int,
    seed: Optional[int],
    save_intermediates: bool
) -> None:
    """Generate artistic image from text using extracted keywords."""
    text_preview = f"{text[:50]}{'...' if len(text) > 50 else ''}"
    click.echo(f"🎨 Generating image from text: '{text_preview}'")
    
    # Initialize pipeline
    pipeline = MojiMosaicPipeline()
    
    # Set output directory
    output_dir = Path(output) if output else Path.cwd() / "moji_output"
    
    # Generate artwork
    with click.progressbar(length=3, label='Processing') as bar:
        result = pipeline.generate_complete_artwork(
            text=text,
            mosaic_size=mosaic_size,
            diffusion_strength=strength,
            num_keywords=keywords,
            seed=seed,
            save_intermediates=save_intermediates,
            output_dir=output_dir,
        )
        bar.update(3)
    
    # Display results
    if result.get('metadata', {}).get('success', False):
        click.echo(f"✅ Generation completed successfully!")
        click.echo(f"📁 Output saved to: {output_dir}")
        click.echo(f"🔑 Keywords extracted: {', '.join(result['intermediate_results']['keywords'])}")
        click.echo(f"⏱️  Total time: {result['timing']['total_time']:.2f}s")
        
        if save_intermediates:
            click.echo(f"🖼️  Mosaic image: {output_dir / '01_mosaic.png'}")
            click.echo(f"🎭 Final artwork: {output_dir / '02_final_artistic.png'}")
    else:
        click.echo(f"❌ Generation failed: {result.get('error', 'Unknown error')}")
        raise click.ClickException("Generation failed")


@main.command()
@click.argument('text', type=str)
@click.option('--variations', '-n', default=3, help='Number of variations to generate')
@click.option('--output', '-o', type=click.Path(), help='Output directory for generated images')
@click.option('--mosaic-size', default=1024, help='Size of intermediate mosaic image')
@click.option('--seed', type=int, help='Base random seed')
def variations(
    text: str,
    variations: int,
    output: Optional[str],
    mosaic_size: int,
    seed: Optional[int]
) -> None:
    """Generate multiple variations from text using different seeds."""
    text_preview = f"{text[:50]}{'...' if len(text) > 50 else ''}"
    click.echo(f"🎨 Generating {variations} variations from text: '{text_preview}'")
    
    # Initialize pipeline
    pipeline = MojiMosaicPipeline()
    
    # Set output directory
    output_dir = Path(output) if output else Path.cwd() / "moji_variations"
    
    # Generate variations
    with click.progressbar(length=variations, label='Generating variations') as bar:
        results = pipeline.generate_variations(
            text=text,
            num_variations=variations,
            mosaic_size=mosaic_size,
            seed=seed,
            save_intermediates=True,
            output_dir=output_dir,
        )
        bar.update(variations)
    
    # Display results
    successful = sum(1 for r in results if r.get('metadata', {}).get('success', False))
    click.echo(f"✅ Generated {successful}/{variations} variations successfully!")
    click.echo(f"📁 Output saved to: {output_dir}")
    
    for i, result in enumerate(results):
        if result.get('metadata', {}).get('success', False):
            seed_used = result['variation_info']['seed_used']
            timing = result['timing']['total_time']
            click.echo(f"  Variation {i+1}: seed {seed_used} ({timing:.1f}s)")


@main.command()
@click.argument('file', type=click.File('r', encoding='utf-8'))
@click.option('--output', '-o', type=click.Path(), help='Output directory for generated images')
def from_file(file, output: Optional[str]) -> None:
    """Generate artistic image from text file using extracted keywords."""
    text = file.read().strip()
    if not text:
        raise click.ClickException("Input file is empty")
    
    click.echo(f"📄 Processing text file with {len(text)} characters")
    
    # Initialize pipeline
    pipeline = MojiMosaicPipeline()
    
    # Set output directory
    output_dir = Path(output) if output else Path.cwd() / "moji_from_file"
    
    # Generate artwork
    result = pipeline.generate_complete_artwork(
        text=text,
        save_intermediates=True,
        output_dir=output_dir,
    )
    
    # Display results
    if result.get('metadata', {}).get('success', False):
        click.echo(f"✅ Generation completed successfully!")
        click.echo(f"📁 Output saved to: {output_dir}")
        click.echo(f"🔑 Keywords: {', '.join(result['intermediate_results']['keywords'])}")
    else:
        click.echo(f"❌ Generation failed: {result.get('error', 'Unknown error')}")


@main.command()
@click.argument('texts', nargs=-1, required=True)
@click.option('--output', '-o', type=click.Path(), help='Base output directory')
def batch(texts: tuple, output: Optional[str]) -> None:
    """Process multiple texts in batch using extracted keywords."""
    click.echo(f"🔄 Processing {len(texts)} texts in batch")
    
    # Initialize pipeline
    pipeline = MojiMosaicPipeline()
    
    # Set output directory
    output_dir = Path(output) if output else Path.cwd() / "moji_batch"
    
    # Process batch
    with click.progressbar(length=len(texts), label='Processing batch') as bar:
        results = pipeline.batch_process(
            texts=list(texts),
            output_base_dir=output_dir,
            save_intermediates=True,
        )
        bar.update(len(texts))
    
    # Display results
    successful = sum(1 for r in results if r.get('metadata', {}).get('success', False))
    click.echo(f"✅ Processed {successful}/{len(texts)} texts successfully!")
    click.echo(f"📁 Output saved to: {output_dir}")


@main.command()
def info() -> None:
    """Show pipeline information and system status."""
    pipeline = MojiMosaicPipeline()
    info_data = pipeline.get_pipeline_info()
    
    click.echo("🔧 Moji-mosaic Pipeline Information")
    click.echo("=" * 40)
    click.echo(f"Version: {info_data['pipeline_version']}")
    click.echo(f"Cache Directory: {info_data['cache_dir']}")
    click.echo()
    
    click.echo("📦 Components:")
    for name, component in info_data['components'].items():
        click.echo(f"  {name}:")
        for key, value in component.items():
            click.echo(f"    {key}: {value}")
        click.echo()


@main.command()
@click.option('--days', default=30, help='Remove cache files older than this many days')
@click.confirmation_option(prompt='Are you sure you want to clean the cache?')
def clean_cache(days: int) -> None:
    """Clean up old cache files."""
    pipeline = MojiMosaicPipeline()
    pipeline.cleanup_cache(older_than_days=days)
    click.echo(f"🧹 Cleaned cache files older than {days} days")


if __name__ == '__main__':
    main()
