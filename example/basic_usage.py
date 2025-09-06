#!/usr/bin/env python3
"""
Basic usage examples for Moji-mosaic 2.0
"""
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent / "src"))

from moji_mosaic import MojiMosaicPipeline


def basic_example():
    """Basic text-to-art generation example."""
    print("🎨 Basic Moji-mosaic Example")
    print("=" * 40)
    
    # Initialize pipeline
    pipeline = MojiMosaicPipeline(
        diffusion_config={"model_name": "stabilityai/stable-diffusion-3.5-medium"}
    )

    # Sample text
    text = (
        "Death Stranding 2, I bought it out of habit at first, but as I kept "
        "playing day by day, I started wanting to play more.\n\n"
        "I'm moving forward at a slow pace."
    )
    
    print(f"📝 Input text: {text}")
    print("🔄 Processing...")
    
    # Generate artwork using only extracted keywords
    result = pipeline.generate_complete_artwork(
        text=text,
        num_keywords=5,
        save_intermediates=True,
        output_dir=Path("./output/basic_example"),
        tile_shape="circle"
    )
    
    # Display results
    if result['metadata']['success']:
        print("✅ Generation successful!")
        print(f"🔑 Keywords: {', '.join(result['intermediate_results']['keywords'])}")
        print(f"⏱️  Total time: {result['timing']['total_time']:.2f}s")
        print(f"📁 Output saved to: ./output/basic_example/")
    else:
        print(f"❌ Generation failed: {result.get('error', 'Unknown error')}")


def seed_comparison():
    """Compare different seeds using the same keywords."""
    print("\n🎲 Seed Comparison Example")
    print("=" * 40)
    
    pipeline = MojiMosaicPipeline()
    text = """# Digital Souls

    In neon worlds where avatars meet,
    Dragon wings and robot feet,
    We slip on headsets, leave behind
    The weight of flesh, the earthbound mind.

    Strangers become the closest friends
    In spaces where reality bends,
    Through digital skin, hearts beat true—
    In VRChat worlds, we're me and you.

    Portal hop from world to world,
    User-crafted dreams unfurled,
    When headsets rest, connections remain—
    We've found new ways to be human."""
    seeds = [42, 123, 456, 789]
    
    print(f"📝 Input text: {text}")
    print(f"🎨 Generating {len(seeds)} variations with different seeds...")
    
    for seed in seeds:
        print(f"\n🔄 Processing with seed {seed}...")
        
        result = pipeline.generate_complete_artwork(
            text=text,
            seed=seed,
            save_intermediates=True,
            output_dir=Path(f"./output/seed_comparison/seed_{seed}")
        )
        
        if result['metadata']['success']:
            keywords = ', '.join(result['intermediate_results']['keywords'][:3])
            print(f"✅ Seed {seed} completed ({result['timing']['total_time']:.1f}s)")
            print(f"   Keywords: {keywords}...")
        else:
            print(f"❌ Seed {seed} failed")


def batch_processing():
    """Batch processing example."""
    print("\n📦 Batch Processing Example")
    print("=" * 40)
    
    pipeline = MojiMosaicPipeline()
    
    texts = [
        "A peaceful garden with blooming flowers",
        "Futuristic city with flying cars",
        "Ancient castle on a misty hill",
        "Deep ocean with mysterious creatures"
    ]
    
    print(f"📝 Processing {len(texts)} texts in batch...")
    
    results = pipeline.batch_process(
        texts=texts,
        save_intermediates=True,
        output_base_dir=Path("./output/batch_processing")
    )
    
    print("\n📊 Batch Results:")
    successful = 0
    for i, result in enumerate(results):
        if result['metadata']['success']:
            successful += 1
            keywords = ', '.join(result['intermediate_results']['keywords'][:3])
            time_taken = result['timing']['total_time']
            print(f"  ✅ Text {i+1}: {keywords}... ({time_taken:.1f}s)")
        else:
            print(f"  ❌ Text {i+1}: Failed")
    
    print(f"\n🎯 Success rate: {successful}/{len(texts)} ({successful/len(texts)*100:.1f}%)")


def variations_example():
    """Generate multiple variations of the same text using different seeds."""
    print("\n🔄 Variations Example")
    print("=" * 40)
    
    pipeline = MojiMosaicPipeline()
    text = """# Digital Souls

    In neon worlds where avatars meet,
    Dragon wings and robot feet,
    We slip on headsets, leave behind
    The weight of flesh, the earthbound mind.

    Strangers become the closest friends
    In spaces where reality bends,
    Through digital skin, hearts beat true—
    In VRChat worlds, we're me and you.

    Portal hop from world to world,
    User-crafted dreams unfurled,
    When headsets rest, connections remain—
    We've found new ways to be human."""
    
    print(f"📝 Input text: {text}")
    print("🎨 Generating 3 variations using different seeds...")
    
    results = pipeline.generate_variations(
        text=text,
        num_variations=3,
        save_intermediates=True,
        output_dir=Path("./output/variations")
    )
    
    print("\n🎭 Variation Results:")
    for i, result in enumerate(results):
        if result['metadata']['success']:
            seed_used = result['variation_info']['seed_used']
            time_taken = result['timing']['total_time']
            keywords = ', '.join(result['intermediate_results']['keywords'][:3])
            print(f"  ✅ Variation {i+1} (seed {seed_used}): {time_taken:.1f}s")
            print(f"      Keywords: {keywords}...")
        else:
            print(f"  ❌ Variation {i+1}: Failed")


def main():
    """Run all examples."""
    print("🚀 Moji-mosaic 2.0 Examples")
    print("=" * 50)
    
    # Create output directory
    Path("./output").mkdir(exist_ok=True)
    
    try:
        # Run examples
        basic_example()
        seed_comparison()
        batch_processing()
        variations_example()
        
        print("\n🎉 All examples completed!")
        print("📁 Check the ./output/ directory for generated images")
        
    except Exception as e:
        print(f"\n❌ Example failed: {e}")
        print("💡 Make sure you have installed all dependencies:")
        print("   pip install -e .")


if __name__ == "__main__":
    main()
