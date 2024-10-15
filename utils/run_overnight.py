import subprocess
import sys
import time
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def prevent_sleep():
    subprocess.Popen(['caffeinate', '-d'])


def run_ml_script():
    sys.path.insert(0, PROJECT_ROOT)

    import app

    app.csic_path = os.path.join(PROJECT_ROOT, 'data', 'csic_database.csv')
    app.cidds_external_path = os.path.join(PROJECT_ROOT, 'data', 'cidds-001-externalserver.parquet')
    app.cidds_openstack_path = os.path.join(PROJECT_ROOT, 'data', 'cidds-001-openstack.parquet')
    app.logs_path = os.path.join(PROJECT_ROOT, 'data', 'logs.json')

    # Run your main function
    app.main()


def main():
    print("Starting overnight ML task...")
    print(f"Project root: {PROJECT_ROOT}")

    prevent_sleep()

    try:
        start_time = time.time()
        run_ml_script()
        end_time = time.time()

        print(f"ML task completed in {end_time - start_time:.2f} seconds")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        subprocess.run(['killall', 'caffeinate'])
        print("Sleep prevention disabled. Script execution complete.")


if __name__ == "__main__":
    main()