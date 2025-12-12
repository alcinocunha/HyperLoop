import sys

if __name__ == "__main__":
    # check command line arguments
    if len(sys.argv) != 3:
        print("Usage: python cms.py <reviewers> <articles>")
        sys.exit(1)

    R = int(sys.argv[1])  # number of reviewers
    A = int(sys.argv[2])  # number of articles

    print("MODULE main")
    print("FROZENVAR")

    
    for a in range(A):
        for r in range(R):
            print(f"    assigns_{a}_{r} : boolean;")

    print("VAR")
    
    # possible reviews: 0 (no review), 1 (reject), 2 (major revision), 3 (accept)
    
    for a in range(A):
        for r in range(R):
            print(f"    review_{a}_{r} : 0..3;")
    
    for a in range(A):
        print(f"    decision_{a} : 0..3;")

    print("INIT")

    constraints = []
    for a in range(A):
        constraints.append("(" + " | ".join([f"assigns_{a}_{r}" for r in range(R)]) + ")")
    for r in range(R):
        constraints.append("(" + " | ".join([f"assigns_{a}_{r}" for a in range(A)]) + ")")
    for a in range(A):
        for r in range(R):
            constraints.append(f"review_{a}_{r} = 0")
    for a in range(A):
        constraints.append(f"decision_{a} = 0")

    print("    "+" &\n    ".join(constraints))

    print("TRANS")

    def stutter():
        stmts = []
        for a in range(A):
            for r in range(R):
                stmts.append(f"next(review_{a}_{r}) = review_{a}_{r}")
        for a in range(A):
            stmts.append(f"next(decision_{a}) = decision_{a}")
        return "("+ " & ".join(stmts) + ")"
    
    def review(a,r):
        stmts = []
        stmts.append(f"assigns_{a}_{r}")
        stmts.append(f"review_{a}_{r} = 0")
        stmts.append(f"next(review_{a}_{r}) != 0")
        for a2 in range(A):
            for r2 in range(R):
                if a2 != a or r2 != r:
                    stmts.append(f"next(review_{a2}_{r2}) = review_{a2}_{r2}")
        for a2 in range(A):
            stmts.append(f"next(decision_{a2}) = decision_{a2}")
        return "(" + " & ".join(stmts) + ")"
    
    def decide(a):
        stmts = []
        stmts.append(f"decision_{a} = 0")
        stmts.append("(" + " & ".join([f"(assigns_{a}_{r} -> review_{a}_{r} != 0)" for r in range(R)]) + ")")
        stmts.append(f"next(decision_{a}) != 0")
        stmts.append("(" + " | ".join([f"next(decision_{a}) = review_{a}_{r}" for r in range(R)]) + ")")
        for a2 in range(A):
            if a2 != a:
                stmts.append(f"next(decision_{a2}) = decision_{a2}")
        for a2 in range(A):
            for r2 in range(R):
                stmts.append(f"next(review_{a2}_{r2}) = review_{a2}_{r2}")
        return "(" + " & ".join(stmts) + ")"

    print("    " + " |\n    ".join([stutter()] + [review(a,r) for a in range(A) for r in range(R)] + [decide(a) for a in range(A)]))

    print("HLTLSPEC")

    def same_assigns(T,U):
        stmts = []
        for a in range(A):
            for r in range(R):
                stmts.append(f"assigns_{a}_{r}[{T}] = assigns_{a}_{r}[{U}]")
        return "(" + " & ".join(stmts) + ")"

    def aligned(r,T,U):
        stmts = []
        for a in range(A):
            stmts.append(f"(assigns_{a}_{r}[{T}] -> (decision_{a}[{T}] = 0 <-> decision_{a}[{U}] = 0) & " + " & ".join([f"(review_{a}_{r2}[{T}] = 0 <-> review_{a}_{r2}[{U}] = 0)" for r2 in range(R)]) + ")")
        return "G (" + " & ".join(stmts) + ")"

    def all_decided(T):
        stmts = []
        for a in range(A):
            stmts.append(f"decision_{a}[{T}] != 0")
        return "F (" + " & ".join(stmts) + ")"

    def same_reviews_decisions(r,T,U):
        stmts = []
        for a in range(A):
            stmts.append(f"(assigns_{a}_{r}[{T}] -> (decision_{a}[{T}] = decision_{a}[{U}] & " + " & ".join([f"review_{a}_{r2}[{T}] = review_{a}_{r2}[{U}]" for r2 in range(R)]) + "))")
        return "G (" + " & ".join(stmts) + ")"
    
    def same_reviews_others(r,T,U):
        stmts = []
        for a in range(A):
            stmts.append(f"(!assigns_{a}_{r}[{T}] -> (" + " & ".join([f"review_{a}_{r2}[{T}] = review_{a}_{r2}[{U}]" for r2 in range(R)]) + "))")
        return "G (" + " & ".join(stmts) + ")"

    print("    forall A . forall B . exists C .\n    (" + same_assigns("A","B") + " & " + aligned(0,"A","B") + " & " + all_decided("A") + ")\n    ->\n    (" + same_assigns("B","C") + " & " + same_reviews_decisions(0,"A","C") + " & " + same_reviews_others(0,"B","C") + ")")