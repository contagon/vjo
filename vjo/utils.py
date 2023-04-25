import seaborn as sns


def setup_plot():
    sns.set_style("whitegrid")
    sns.set_palette("deep")
    sns.set_context("notebook")
    return sns.color_palette("deep")
