from matplotlib.lines import Line2D
from sklearn.manifold import MDS
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import jinja2
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import os

from ..dataset import Photo
from ..scraper import ImageScraper


def _output(output_path):
    if output_path is None:
        plt.show()
    else:
        plt.savefig(output_path, format="pdf")
        plt.clf()


def plot_match_nonmatch_confidences(
    provider_enum, provider_aggregate_metrics, output_path=None
):
    # https://stackoverflow.com/questions/32899463/how-can-i-overlay-two-graphs-in-seaborn
    fig, ax = plt.subplots()
    sns.kdeplot(
        provider_aggregate_metrics["no_match_confidences"], ax=ax, label="No Match"
    )
    sns.kdeplot(
        provider_aggregate_metrics["true_match_confidences"], ax=ax, label="Match"
    )
    plt.xlabel("Confidence")
    plt.ylabel("P(Confidence)")
    # TODO - change the provider enum id to the provider str label.
    plt.title(f"P(Confidence | Match, No Match) for Provider #{provider_enum.value}")
    plt.legend(title="Ground Truth")
    _output(output_path)


def plot_all_confidences(provider_enum, provider_aggregate_metrics, output_path=None):
    sns.kdeplot(provider_aggregate_metrics["all_confidences"])
    plt.xlabel("Confidence")
    plt.ylabel("P(Confidence)")
    # TODO - change the provider enum id to the provider str label.
    plt.title(f"P(Confidence) for Provider #{provider_enum.value}")
    _output(output_path)


def plot_fmr_fnmr(provider_enum, provider_aggregate_metrics, output_path=None):
    if provider_aggregate_metrics["false_match_rate"] is None:
        print("warning - plot_fmr_fnmr skipped due to empty input")
        return
    # x, y
    plt.plot(
        provider_aggregate_metrics["false_match_rate"],
        provider_aggregate_metrics["false_non_match_rate"],
    )
    plt.xlabel("False Match Rate")
    plt.ylabel("False Non Match Rate")
    plt.xscale("log")
    plt.yscale("log")
    # TODO - change the provider enum id to the provider str label.
    plt.title(f"FNMR vs FMR for Provider #{provider_enum.value}")
    _output(output_path)


def plot_many_fmr_fnmr(all_results, output_path=None):
    # x, y
    colors = list(mcolors.TABLEAU_COLORS.keys())
    for i, (provider_enum, provider_results) in enumerate(all_results.items()):
        provider_aggregate_metrics = provider_results["metrics"]["aggregate"]
        plt.plot(
            provider_aggregate_metrics["false_match_rate"],
            provider_aggregate_metrics["false_non_match_rate"],
            color=colors[i],
            # TODO - change the provider enum id to the provider str label.
            label=f"Provider #{provider_enum.value}",
        )
    plt.xlabel("False Match Rate")
    plt.ylabel("False Non Match Rate")
    plt.xscale("log")
    plt.yscale("log")
    plt.title(f"FNMR vs FMR for All Providers")
    plt.legend()
    _output(output_path)


def create_extreme_faces_landing_page(provider_enum, names_ids, output_path):
    # https://stackoverflow.com/a/38642558
    templateDir = os.sep.join(__file__.split(os.sep)[:-1]) + os.sep + "templates"
    templateLoader = jinja2.FileSystemLoader(searchpath=templateDir)
    templateEnv = jinja2.Environment(loader=templateLoader)

    TEMPLATE_FILE = "results_home.html"
    template = templateEnv.get_template(TEMPLATE_FILE)

    outputText = template.render(
        names_ids=names_ids,
        provider_id=provider_enum.value,
    )

    with open(output_path, "w") as f:
        f.write(outputText)


def show_extreme_faces(
    database,
    faces,
    people,
    photo_id_to_url,
    provider_enum,
    metrics,
    same_id,
    output_path,
    avg_or_median,
    confusing_or_representative,
    multiple_ids,
    N=15,
):
    N = min(N, len(metrics[avg_or_median]))

    tuples = []

    if confusing_or_representative == "confusing":
        extreme_results = metrics[avg_or_median][:N]
    elif confusing_or_representative == "representative":
        extreme_results = metrics[avg_or_median][-N:][::-1]

    for (avg_confidence, person_id, face_id, count) in extreme_results:
        bbox = faces[face_id].bounding_box.get_top_left_width_height()
        photo_id = faces[face_id].photo_id
        url = photo_id_to_url[photo_id]
        if Photo._needs_to_be_hosted(url):
            # TODO - check this for bugs.
            url = (
                "'"
                + os.getcwd()
                + os.sep
                + ImageScraper.get_image_filename(database.get_photo_dir(), photo_id)
                + "'"
            )
        name = people[person_id].name
        int_confidence = int(100 * avg_confidence)
        tuples.append((f"{int_confidence}%", name, face_id, url, bbox, count))

    # https://stackoverflow.com/a/38642558
    templateDir = os.sep.join(__file__.split(os.sep)[:-1]) + os.sep + "templates"
    templateLoader = jinja2.FileSystemLoader(searchpath=templateDir)
    templateEnv = jinja2.Environment(loader=templateLoader)
    # TODO - add a flag to toggle between these!
    # TEMPLATE_FILE = "results.html"
    TEMPLATE_FILE = "cropped_results.html"
    template = templateEnv.get_template(TEMPLATE_FILE)

    conf_id_url_bbox = list(enumerate(tuples))

    label = confusing_or_representative
    if confusing_or_representative == "representative" and not same_id:
        label = "distinguishable"

    outputText = template.render(
        conf_id_url_bbox=conf_id_url_bbox,
        num_faces=len(conf_id_url_bbox),
        same_id=same_id,
        avg_or_median=avg_or_median,
        confusing_or_representative=label,
        multiple_ids=multiple_ids,
    )

    with open(output_path, "w") as f:
        f.write(outputText)


def show_extreme_face_pairs(
    database,
    faces,
    people,
    photo_id_to_url,
    provider_enum,
    metrics,
    same_id,
    output_path,
    confusing_or_representative,
    multiple_ids,
    N=15,
):
    N = min(N, len(metrics["raw"]))

    tuples = []

    if confusing_or_representative == "confusing":
        extreme_results = metrics["raw"][:N]
    elif confusing_or_representative == "representative":
        extreme_results = metrics["raw"][-N:][::-1]

    if same_id:
        iterable = []
        for (confidence, person, face1_id, face2_id) in extreme_results:
            iterable.append((confidence, ((face1_id, person), (face2_id, person))))
    else:
        iterable = extreme_results

    for (confidence, ((face1_id, person1_id), (face2_id, person2_id))) in iterable:
        bbox1 = faces[face1_id].bounding_box.get_top_left_width_height()
        photo1_id = faces[face1_id].photo_id
        url1 = photo_id_to_url[photo1_id]
        if Photo._needs_to_be_hosted(url1):
            url1 = (
                "'"
                + os.getcwd()
                + os.sep
                + ImageScraper.get_image_filename(database.get_photo_dir(), photo1_id)
                + "'"
            )
        name1 = people[person1_id].name

        bbox2 = faces[face2_id].bounding_box.get_top_left_width_height()
        photo2_id = faces[face2_id].photo_id
        url2 = photo_id_to_url[photo2_id]
        if Photo._needs_to_be_hosted(url2):
            url2 = (
                "'"
                + os.getcwd()
                + os.sep
                + ImageScraper.get_image_filename(database.get_photo_dir(), photo2_id)
                + "'"
            )
        name2 = people[person2_id].name

        int_confidence = int(100 * confidence)
        tuples.append(
            (
                f"{int_confidence}%",
                name1,
                name2,
                face1_id,
                face2_id,
                url1,
                url2,
                bbox1,
                bbox2,
            )
        )

    # https://stackoverflow.com/a/38642558
    templateDir = os.sep.join(__file__.split(os.sep)[:-1]) + os.sep + "templates"
    templateLoader = jinja2.FileSystemLoader(searchpath=templateDir)
    templateEnv = jinja2.Environment(loader=templateLoader)
    # TODO - add a flag to toggle between these!
    # TEMPLATE_FILE = "results_pairs.html"
    TEMPLATE_FILE = "cropped_results_pairs.html"
    template = templateEnv.get_template(TEMPLATE_FILE)

    name_url_bbox_pairs = list(enumerate(tuples))

    label = confusing_or_representative
    if confusing_or_representative == "representative" and not same_id:
        label = "distinguishable"

    outputText = template.render(
        name_url_bbox_pairs=name_url_bbox_pairs,
        num_pairs=len(name_url_bbox_pairs),
        same_id=same_id,
        confusing_or_representative=label,
        multiple_ids=multiple_ids,
    )

    with open(output_path, "w") as f:
        f.write(outputText)


def show_faces_2d_embedding(
    database, matrix, face_to_person, face_id_to_face, show_faces=False, out=None
):
    if matrix["data"] is None or len(matrix) < 2:
        return

    # https://stackabuse.com/guide-to-multidimensional-scaling-in-python-with-scikit-learn/
    mds = MDS(n_components=2, dissimilarity="precomputed")

    # Convert similarity into dissimilarity by taking the difference with 1.
    data = 1 - matrix["data"]

    # Get the MDS embeddings.
    pts = mds.fit_transform(data)

    if out is None:
        out_w_faces = None
        out_w_points = None
    else:
        out_w_faces = out.replace("REPLACE", "img")
        out_w_points = out.replace("REPLACE", "scatter")

    show_faces_2d_embedding_helper(
        database,
        pts,
        matrix,
        face_to_person,
        face_id_to_face,
        show_faces=True,
        out=out_w_faces,
    )
    show_faces_2d_embedding_helper(
        database,
        pts,
        matrix,
        face_to_person,
        face_id_to_face,
        show_faces=False,
        out=out_w_points,
    )


def show_faces_2d_embedding_helper(
    database, pts, matrix, face_to_person, face_id_to_face, show_faces=False, out=None
):
    if show_faces:
        face_id_to_box = {}
        for face_id, face in face_id_to_face.items():
            try:
                img = face.crop(database, skip_padding=True, return_image=True)
                img = img.resize((80, 80))
            except:
                img = None
            face_id_to_box[face_id] = img

    person_ids = [face_to_person[i] for i in matrix["ids"]]
    person_names = [
        database.get_person(person_id).get_name() for person_id in person_ids
    ]

    # Plot the embedding, colored according to the class of the points
    ax = sns.scatterplot(x=pts[:, 0], y=pts[:, 1], hue=person_names)

    if show_faces:
        ax.get_legend().remove()

        for i, (face_id, [x, y]) in enumerate(zip(matrix["ids"], pts)):
            img = face_id_to_box[face_id]
            try:
                imagebox = OffsetImage(img, zoom=0.3, cmap=plt.cm.gray)
            except Exception as e:
                print("Exception in show_faces_2d_embedding_helper", e)
                continue
            ab = AnnotationBbox(imagebox, (x, y), frameon=False)
            ax.add_artist(ab)

    # TODO - add provider name into output title??
    if len(set(person_names)) == 1:
        person = list(set(person_names))[0]
        ax.set_title(
            f"MDS on Graph of Face-Compare Confidences for {person}", fontsize=12
        )
    else:
        ax.set_title("MDS on Graph of Face-Compare Confidences", fontsize=14)

    _output(out)


def show_dataset_statistics(
    metadata, people, out_base, groups=None, disjoint_groups=None
):
    print(f'The dataset had {metadata["num_identities"]} seed identities.')

    num_comparisons = metadata["num_comparisons"]
    print(
        f'The labeled dataset was used to generate {num_comparisons["labeled-same-id"]} same-id comparisons.'
    )
    print(
        f'The labeled dataset was used to generate {num_comparisons["labeled-diff-id"]} diff-id comparisons.'
    )
    print(
        f'The unlabeled dataset was used to generate {num_comparisons["unlabeled-same-seed"]} same-seed-id comparisons.'
    )
    print(
        f'The unlabeled dataset was used to generate {num_comparisons["unlabeled-diff-seed"]} diff-seed-id comparisons.'
    )

    # 1. Plot histogram of number of single face images (per identity)
    fig, ax = plt.subplots()
    total_faces = list(metadata["total_faces_per_id"].values())
    hist = ax.hist(total_faces, bins=[5 * i for i in range(21)], rwidth=0.9, alpha=0.5)
    ax.set_title("Number total faces collected per seed id")
    ax.set_xlabel("Number of faces")
    ax.set_ylabel("Number of seed identies")
    median_t = np.median(total_faces)
    if disjoint_groups is not None:
        colors = plt.cm.tab10.colors
        plot_height = max(hist[0])
        arrow_base = plot_height / 8
        arrow_specs = dict(width=0.25, head_width=1, head_length=0.45 * arrow_base)
        ax.arrow(
            median_t,
            arrow_base,
            0,
            -0.5 * arrow_base,
            color=colors[1],
            label=f"Overall ({median_t:0.1f})",
            **arrow_specs,
        )
        for i, label in enumerate(disjoint_groups):
            total_faces_per_group = [
                v
                for k, v in metadata["total_faces_per_id"].items()
                if k in groups[label]
            ]
            group_median_t = np.median(total_faces_per_group)
            ax.arrow(
                group_median_t,
                arrow_base,
                0,
                -0.5 * arrow_base,
                label=f"{label} ({group_median_t:0.1f})",
                color=colors[2 + i],
                **arrow_specs,
            )
        ax.legend(title="Median per group")
        _output(out_base + "total_faces_by_group.pdf")
    else:
        mean_t = np.mean(total_faces)
        ax.vlines(
            mean_t,
            0,
            1,
            transform=ax.get_xaxis_transform(),
            color="red",
            label=f"mean ({mean_t:0.1f})",
        )
        ax.vlines(
            median_t,
            0,
            1,
            transform=ax.get_xaxis_transform(),
            color="black",
            label=f"median ({median_t:0.1f})",
        )
        ax.legend()
        _output(out_base + "total_faces.pdf")

    # 2. Plot histogram of % correct id faces (per identity)
    fig, ax = plt.subplots()
    correct_faces = list(metadata["correct_faces_percentages_per_id"].values())
    if set(correct_faces) == {None}:
        return
    hist = ax.hist(
        correct_faces, bins=[5 * i for i in range(21)], rwidth=0.9, alpha=0.5
    )
    ax.set_title("Percentage of faces matching seed id")
    ax.set_xlabel("Percentage of faces")
    ax.set_ylabel("Number of seed identies")
    median_c = np.median(correct_faces)
    if disjoint_groups is not None:
        plot_height = max(hist[0])
        arrow_base = plot_height / 8
        arrow_specs = dict(width=0.25, head_width=1, head_length=0.45 * arrow_base)
        ax.arrow(
            median_c,
            arrow_base,
            0,
            -0.5 * arrow_base,
            color=colors[1],
            label=f"Overall ({median_c:0.1f})",
            **arrow_specs,
        )
        for i, label in enumerate(disjoint_groups):
            correct_faces_per_group = [
                v
                for k, v in metadata["correct_faces_percentages_per_id"].items()
                if k in groups[label]
            ]
            group_median_c = np.median(correct_faces_per_group)
            ax.arrow(
                group_median_c,
                arrow_base,
                0,
                -0.5 * arrow_base,
                color=colors[2 + i],
                label=f"{label} ({group_median_c:0.1f})",
                **arrow_specs,
            )
        ax.legend(title="Median per group")
        _output(out_base + "correct_faces_by_group.pdf")
    else:
        mean_c = np.mean(correct_faces)
        ax.vlines(
            mean_c,
            0,
            1,
            transform=ax.get_xaxis_transform(),
            color="red",
            label=f"mean ({mean_c:0.1f})",
        )
        ax.vlines(
            median_c,
            0,
            1,
            transform=ax.get_xaxis_transform(),
            color="black",
            label=f"median ({median_c:0.1f})",
        )
        ax.legend()
        _output(out_base + "correct_faces.pdf")

    # 3. Scatterplot of # of single id faces against % correct id faces
    if disjoint_groups is not None:
        name_to_group = {}
        for i, group in enumerate(disjoint_groups):
            for id in groups[group]:
                name_to_group[people[id].name] = i
    fig, ax = plt.subplots()
    assert (
        metadata["total_faces_per_id"].keys()
        == metadata["correct_faces_percentages_per_id"].keys()
    )
    names = [people[key].name for key in metadata["total_faces_per_id"]]
    for i, name in enumerate(names):
        if disjoint_groups is not None:
            color_arg = dict(color=colors[name_to_group[name]])
        else:
            color_arg = dict()
        plt.annotate(
            name.replace(" ", "\n"),
            (total_faces[i], 1 + correct_faces[i]),
            ha="center",
            fontsize=4,
        )
        ax.scatter(total_faces[i], correct_faces[i], **color_arg)
    ax.set_title("Percentage of faces matching seed id vs total number of faces")
    ax.set_xlabel("Total number of faces")
    ax.set_ylabel("Percentage of faces matching seed id")
    ax.set_xbound(min(total_faces) - 10, max(total_faces) + 10)
    ax.set_ybound(min(correct_faces) - 10, 100)
    fig.set_size_inches(8, 6)
    if disjoint_groups is not None:
        ax.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label=disjoint_groups[i],
                    markerfacecolor=colors[i],
                    markersize=10,
                )
                for i in range(len(disjoint_groups))
            ]
        )
        _output(out_base + "total_vs_correct_faces_by_group.pdf")
    else:
        _output(out_base + "total_vs_correct_faces.pdf")
