import sys

if __name__ == "__main__":
    # check command line arguments
    if len(sys.argv) != 2:
        print("Usage: python bakery.py <processes>")
        sys.exit(1)

    N = int(sys.argv[1])  # number of processes

    print("Forall A . !(" + " & ".join([f"G (F (pc_{i}[A] = 4))" for i in range(N)]) + ")")