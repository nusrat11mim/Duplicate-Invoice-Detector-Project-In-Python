import os


def main():
    if not os.path.exists("invoices.csv"):
        print("No invoices.csv found -> generating sample data...\n")
        import generate_sample_data
        generate_sample_data.main()
        print()

    if not os.path.exists("invoice_dup_model.joblib"):
        print("No trained model found -> training model...\n")
        import train_model
        train_model.main()
        print()

    print("Running duplicate detection...\n")
    import detect_duplicates
    detect_duplicates.main()


if __name__ == "__main__":
    main()
