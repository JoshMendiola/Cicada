import subprocess
import sys
import time


def prevent_sleep():
    # This function will run caffeinate in the background
    subprocess.Popen(['caffeinate', '-d'])


def run_ml_script():
    import app
    app.main()


def main():
    print("Starting overnight ML task...")

    # Prevent sleep
    prevent_sleep()

    try:
        # Run your ML script
        start_time = time.time()
        run_ml_script()
        end_time = time.time()

        print(f"ML task completed in {end_time - start_time:.2f} seconds")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Stop caffeinate
        subprocess.run(['killall', 'caffeinate'])
        print("Sleep prevention disabled. Script execution complete.")


if __name__ == "__main__":
    main()