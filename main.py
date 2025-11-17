"""

"""

from scripts import (
    stock_assessments,
    species_landings,
    sofia_landings,
    stock_weights,
    stock_landings,
    aggregate_tables,
    capture_production_figures,
)

if __name__ == "__main__":
    print("Creating stock assessments lists...")

    stock_assessments.main()

    print("Computing species landings...")

    species_landings.main()

    print("Computing SOFIA landings...")

    sofia_landings.main()

    print("Computing stock weights...")

    stock_weights.main()

    print("Computing stock landings...")

    stock_landings.main()

    print("Computing aggregate tables...")

    aggregate_tables.main()

    print("Creating capture production figures...")

    capture_production_figures.main()

    print("All scripts successfully run.")
