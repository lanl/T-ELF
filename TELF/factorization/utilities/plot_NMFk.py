import matplotlib.pyplot as plt
import numpy as np
import os 

def plot_RESCALk(data, k_predict, name, path, plot_predict=False, plot_final=False, simple_plot=False, calculate_error=True):

    fig, ax1 = plt.subplots(figsize=(8, 8), dpi=80)
    
    # silhouette
    color = "tab:red"
    ax1.set_xlabel("latent dimension")
    ax1.set_ylabel("silhouette", color=color)

    # sill  
    ax1.plot(
        list(data["Ks"]),
        data["sils_min"],
        "o-",
        color=color,
        label="W minimum silhouette",
    )
    
    # add a vertical line for xtick k-values to make it easier to see which point corresponds to which k
    if not isinstance(data["sils_min"], np.float64):  # if plot contains more than one k-value
        for xtick in ax1.get_xticks():
            if xtick in data["Ks"]:
                y = data["sils_min"][np.where(data["Ks"] == xtick)[0][0]]  # get the y value that corresponds to xtick
                plt.vlines(xtick, min([0, np.min(data["sils_min"])]), y, colors='black', alpha=0.4)

    if not simple_plot:
        ax1.errorbar(
            list(data["Ks"]),
            data["sils_mean"],
            yerr=data["sils_std"],
            fmt="^:",
            color="tab:green",
            label=r"W mean +- std silhouette",
        )

    ax1.set_ylim(min([0, np.min(data["sils_min"])]), 1)
    ax1.tick_params(axis="y", labelcolor=color)
    
    # k prediction
    if plot_predict:
            ax1.axvline(
                x=list(data["Ks"])[list(data["Ks"]).index(k_predict)],
                lw=5,
                alpha=0.8,
                color="lightsteelblue",
                label="K Predict= " + str(k_predict),
            )
        
    # legend
    ax1.legend(
        loc="upper left",
        bbox_to_anchor=(0.5, -0.07),
        fancybox=True,
        shadow=True,
    )
    
    # relative error
    if calculate_error:
        ax2 = ax1.twinx()
        color = "tab:blue"
        ax2.set_ylabel("relative error", color=color)
        ax2.plot(
            list(data["Ks"]),
            data["err_reg"],
            "o-",
            color=color,
            label="regression relative error",
        )
        
        if not simple_plot:
            ax2.errorbar(
                list(data["Ks"]),
                data["err_mean"],
                yerr=data["err_std"],
                fmt="^:",
                color="tab:orange",
                label="perturbation relative error mean +- std",
            )

        ax2.tick_params(axis="y", labelcolor=color)
        ax2.legend(
            loc="upper right",
            bbox_to_anchor=(0.5, -0.07),
            fancybox=True,
            shadow=True,
            handlelength=4,
        )

    ax1.legend(
        loc="upper right",
        bbox_to_anchor=(.9, -0.07),
        fancybox=True,
        shadow=True,
        handlelength=4,
    )
    
    # finalize
    fig.tight_layout()
    plt.title(name + "  " + str(min(list(data["Ks"]))) + "-" + str(max(list(data["Ks"]))))
    plt.ioff()
    
    # save
    if plot_final:
        plt.savefig(
            os.path.join(f'{path}', f'FINAL_k={min(list(data["Ks"]))}-{max(list(data["Ks"]))}.png'),
            bbox_inches="tight",
        )
    else:
        plt.savefig(
            os.path.join(f'{path}', f'k={min(list(data["Ks"]))}-{max(list(data["Ks"]))}.png'),
            bbox_inches="tight",
        )
    plt.close("all")



def plot_SymNMFk(data, name, path, plot_final=False):
    pac = False
    if "pac" in data and len(data["pac"]) > 0:
        pac = True
    
    fig, ax1 = plt.subplots(figsize=(8, 8), dpi=80)
    
    # silhouette
    color = "black"
    ax1.set_xlabel("latent dimension")
    ax1.set_ylabel("PAC", color=color)
    
    # add a vertical line for xtick k-values to make it easier to see which point corresponds to which k
    if not isinstance(data["pac"], np.float64):  # if plot contains more than one k-value
        for xtick in ax1.get_xticks():
            if xtick in data["Ks"]:
                y = data["pac"][np.where(data["Ks"] == xtick)[0][0]]  # get the y value that corresponds to xtick
                plt.vlines(xtick, min(data["pac"] + [0]), y, colors='black', alpha=0.4)

    # pac
    color = "black"
    ax1.plot(
        list(data["Ks"]),
        data["pac"],
        linestyle="--",
        marker="s",
        color=color,
        label="PAC",
    )
    ax1.set_ylim(min(0, np.min(data["pac"])), 1)
    ax1.tick_params(axis="y", labelcolor=color)
        
    # legend
    ax1.legend(
        loc="upper left",
        bbox_to_anchor=(0.5, -0.07),
        fancybox=True,
        shadow=True,
    )
    
    # relative error
    ax2 = ax1.twinx()
    if set(data["err_mean"]) != set([0.0]):
        color = "tab:blue"
        ax2.set_ylabel("Objective", color=color)
        ax2.plot(
            list(data["Ks"]),
            data["err_mean"],
            "o-",
            color=color,
            label="Objective",
        )

    ax2.tick_params(axis="y", labelcolor=color)
    ax2.legend(
        loc="upper right",
        bbox_to_anchor=(0.5, -0.07),
        fancybox=True,
        shadow=True,
    )
    
    # finalize
    fig.tight_layout()
    plt.title(name + "  " + str(min(list(data["Ks"]))) + "-" + str(max(list(data["Ks"]))))
    plt.ioff()
    
    # save
    if plot_final:
        plt.savefig(
            os.path.join(f'{path}', f'FINAL_k={min(list(data["Ks"]))}-{max(list(data["Ks"]))}.png'),
            bbox_inches="tight",
        )
    else:
        plt.savefig(
            os.path.join(f'{path}', f'k={min(list(data["Ks"]))}-{max(list(data["Ks"]))}.png'),
            bbox_inches="tight",
        )
    plt.close("all")


def plot_BNMFk(Ks, sils, bool_err, path=None, name=None):
    fig, ax1 = plt.subplots(figsize=(8, 8), dpi=80)
    plt.rcParams['svg.fonttype'] = 'none'
    color = 'tab:red'
    ax1.set_xlabel('latent dimension')
    ax1.set_ylabel('silhouette', color=color)
    ax1.plot(Ks, sils, 'o-', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('boolean relative error', color=color)
    ax2.plot(Ks, bool_err, '^-', color=color, label='bool_err')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title(str(name) + "  " + str(min(list(Ks))) + "-" + str(max(list(Ks))))
    fig.tight_layout()

    plt.ioff()
    if path is not None:
        plt.savefig(
            os.path.join(f'{path}', f'k={min(list(Ks))}-{max(list(Ks))}.png'),
            bbox_inches="tight",
        )

    plt.close("all")
    return None


def plot_NMFk(
        data, 
        k_predict, 
        name, path, 
        plot_predict=False, 
        plot_final=False,
        simple_plot=False, 
        calculate_error=True,
        Ks_not_computed=[]
        ):

    pac = False
    if "pac" in data and len(data["pac"]) > 0:
        pac = True
    
    fig, ax1 = plt.subplots(figsize=(12, 8), dpi=100)
    
    # silhouette
    color = "tab:red"
    ax1.set_xlabel("latent dimension")
    
    if pac:
        ax1.set_ylabel("silhouette - pac", color="black")
    else:
        ax1.set_ylabel("silhouette", color=color)

    # W sill  
    ax1.plot(
        list(data["Ks"]),
        data["sils_min_W"],
        "o-",
        color=color,
        label="W minimum silhouette",
    )

    # H sill
    ax1.plot(
        list(data["Ks"]),
        data["sils_min_H"],
        "o-.",
        color="purple",
        label="H minimum silhouette",
    )
    
    # add a vertical line for xtick k-values to make it easier to see which point corresponds to which k
    if not isinstance(data["sils_min_W"], np.float64):  # if plot contains more than one k-value
        for xtick in ax1.get_xticks():
            if xtick in data["Ks"]:
                y = data["sils_min_W"][np.where(data["Ks"] == xtick)[0][0]]  # get the y value that corresponds to xtick
                plt.vlines(xtick, min([0, np.min(data["sils_min_H"]), np.min(data["sils_min_W"])]), y, colors='black', alpha=0.4)

    for empty_k in Ks_not_computed:
        plt.vlines(empty_k, ymin=min([0, np.min(data["sils_min_H"]), np.min(data["sils_min_W"])]), ymax=1, colors='black', linewidth=2, alpha=0.9)

    if not simple_plot:
        ax1.errorbar(
            list(data["Ks"]),
            data["sils_mean_W"],
            yerr=data["sils_std_W"],
            fmt="^:",
            color="tab:green",
            label=r"W mean +- std silhouette",
        )

        ax1.errorbar(
            list(data["Ks"]),
            data["sils_mean_H"],
            yerr=data["sils_std_H"],
            fmt="^-.",
            color="tab:green",
            label=r"H mean +- std silhouette",
        )

    # pac
    if pac:
        color = "black"
        ax1.plot(
            list(data["Ks"]),
            data["pac"],
            linestyle="--",
            marker="s",
            color=color,
            label="PAC",
        )
        
        ax1.set_ylim(min([0, np.min(data["sils_min_H"]), np.min(data["sils_min_W"]), np.min(data["pac"])]), 1)
    else:
        ax1.set_ylim(min([0, np.min(data["sils_min_H"]), np.min(data["sils_min_W"])]), 1)
    
    ax1.tick_params(axis="y", labelcolor=color)
    
    # k prediction
    if plot_predict:
            ax1.axvline(
                x=list(data["Ks"])[list(data["Ks"]).index(k_predict)],
                lw=5,
                alpha=0.8,
                color="lightsteelblue",
                label="K Predict= " + str(k_predict),
            )
        
    # legend
    ax1.legend(
        loc="upper left",
        bbox_to_anchor=(0.5, -0.07),
        fancybox=True,
        shadow=True,
    )
    
    # relative error
    if calculate_error:
        ax2 = ax1.twinx()
        color = "tab:blue"
        ax2.set_ylabel("relative error", color=color)
        ax2.plot(
            list(data["Ks"]),
            data["err_reg"],
            "o-",
            color=color,
            label="regression relative error",
        )
        
        if not simple_plot:
            ax2.errorbar(
                list(data["Ks"]),
                data["err_mean"],
                yerr=data["err_std"],
                fmt="^:",
                color="tab:orange",
                label="perturbation relative error mean +- std",
            )

        ax2.tick_params(axis="y", labelcolor=color)
        ax2.legend(
            loc="upper right",
            bbox_to_anchor=(0.5, -0.07),
            fancybox=True,
            shadow=True,
            handlelength=4,
        )

    ax1.legend(
        loc="upper right",
        bbox_to_anchor=(.9, -0.07),
        fancybox=True,
        shadow=True,
        handlelength=4,
    )
    
    
    
    # finalize
    fig.tight_layout()
    plt.title(name + "  " + str(min(list(data["Ks"]))) + "-" + str(max(list(data["Ks"]))))
    plt.ioff()
    
    # save
    if plot_final:
        if len(Ks_not_computed) > 0:
            kmin = min(min(list(data["Ks"])), min(list(Ks_not_computed)))
            kmax = max(max(list(data["Ks"])), max(list(Ks_not_computed)))
        else:
            kmin = min(list(data["Ks"]))
            kmax = max(list(data["Ks"]))

        plt.savefig(
            os.path.join(f'{path}', f'FINAL_k={kmin}-{kmax}.png'),
            bbox_inches="tight",
        )
    else:
        plt.savefig(
            os.path.join(f'{path}', f'k={min(list(data["Ks"]))}-{max(list(data["Ks"]))}.png'),
            bbox_inches="tight",
        )
    plt.close("all")


def plot_consensus_mat(C, figname):  
    
    plt.figure()
    
    try:
        plt.imshow(C)
        plt.colorbar()
        plt.savefig(figname)
    except:
        pass
    
    plt.close("all")


def plot_cophenetic_coeff(Ks, coeff, figname):
    plt.figure()
    plt.plot(Ks, coeff)
    plt.xlabel('k')
    plt.ylabel('Cophenetic Correlation')
    plt.savefig(figname)
    plt.close("all")
