import os
import asyncio
import ffmpeg
import concurrent.futures

def convert_webm_to_mp3(input_path, output_path):
    """
    Converts a single .webm file to .mp3 using ffmpeg-python.
    """
    try:
        ffmpeg.input(input_path).output(output_path).run(quiet=True)
        print(f"Successfully converted {input_path} to {output_path}")
    except ffmpeg.Error as e:
        print(f"Error converting {input_path}:")
        print(e.stderr.decode())

async def process_folder(input_folder):
    """
    Processes all .webm files in the input folder and converts them to .mp3
    concurrently, saving the .mp3 files to a subfolder named 'mp3' within the
    input folder.
    """
    output_folder = os.path.join(input_folder, "mp3")  # Create 'mp3' subfolder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        loop = asyncio.get_event_loop()
        tasks = []
        for filename in os.listdir(input_folder):
            if filename.endswith(".webm"):
                input_path = os.path.join(input_folder, filename)
                output_path = os.path.join(output_folder, os.path.splitext(filename)[0] + ".mp3")
                # Use loop.run_in_executor to run the blocking function in a thread
                task = loop.run_in_executor(executor, convert_webm_to_mp3, input_path, output_path)
                tasks.append(task)

        # Wait for all conversion tasks to complete
        await asyncio.gather(*tasks)

if __name__ == "__main__":
    input_folder = "songs"  # Replace with the path to your .webm files

    async def main():
        await process_folder(input_folder)

    asyncio.run(main())