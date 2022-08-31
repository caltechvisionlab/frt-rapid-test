import os
import sys
from io import StringIO
import datetime
import yaml
import pickle

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from tqdm import tqdm
from PIL import Image

from cfro.analyzer.data_interface import AnalysisDataInterface
from cfro.analyzer.utils.affinity_methods import calculate_confidence_matrix, calculate_rank1_approximation
from cfro.analyzer.utils.vis.matrix import *

import warnings
warnings.filterwarnings("ignore")

STANDARD_FIG_SIZE = (15.4, 10)
FRONT_PAGE_FONTS = 18
DPI = 100

HIST_OVERLAP_RULE = "layer"  # "dodge"

REDIRECT_STDOUT = True


def join_photo_and_query_ids_to_matches(matches_df: pd.DataFrame, detections_df: pd.DataFrame):
    ### join person_id and photo_id to matches_df
    detections_df.set_index("bbox_id", inplace=True)
    matches_df.set_index("bbox0_id", inplace=True)
    matches_df["bbox0_photo_id"] = detections_df["photo_id"].astype("Int64")
    matches_df["bbox0_face_id"] = detections_df["face_id"]
    matches_df["bbox0_query_person_id"] = detections_df["query_person_id"].astype("Int64")
    matches_df.reset_index(inplace=True)
    matches_df.set_index("bbox1_id", inplace=True)
    matches_df["bbox1_photo_id"] = detections_df["photo_id"].astype("Int64")
    matches_df["bbox1_face_id"] = detections_df["face_id"]
    matches_df["bbox1_query_person_id"] = detections_df["query_person_id"].astype("Int64")
    matches_df.reset_index(inplace=True)
    return matches_df


# Class to organize and analyze the confidence data from one service
class ServiceTest:

    def __init__(self, provider_id: int, provider_name: str,
                 matches_df: pd.DataFrame, detections_df_long: pd.DataFrame, image_function
                 ):
        '''Read the data from the file'''

        self.provider_name = provider_name
        self.provider_id = provider_id
        self.get_image = image_function

        self.detections_df_long = detections_df_long

        self.matches_df_full = matches_df.copy()  # same for all providers
        self.matches_df = matches_df[matches_df["provider_id"] == self.provider_id]

        ## sanity checking
        unmatchable_bbox0_ids = self.matches_df[self.matches_df["bbox0_photo_id"].isna()]["bbox0_id"].unique()
        unmatchable_bbox1_ids = self.matches_df[self.matches_df["bbox1_photo_id"].isna()]["bbox1_id"].unique()
        if len(unmatchable_bbox0_ids) + len(unmatchable_bbox1_ids) > 0:
            print(f"Warning: bbox_ids that cannot be matched to a photo_id, provider {provider_id}: "
                  f"{set(unmatchable_bbox0_ids).union(set(unmatchable_bbox1_ids))}")
            self.matches_df.dropna(inplace=True)
        ###

        self.q_person_ids = self.matches_df[
            "bbox0_query_person_id"].unique().tolist()  # set of face IDs -- these are peculiar to the service
        self.n_query_ids = len(self.q_person_ids)  # set of person IDs -- these are the same across services

        # allocate lists that will contain the confidence matrices (original, regularized) and the membership vectors z -- one per identity
        self.C_dict = {}
        self.C1_dict = {}
        self.z_dict = {}
        self.z_list = []
        self.face_ids_dict = {}
        self.ev_ratios = {}

        # identify pairs of images that are associated to the same identity.
        # extract the confidence value that was computed by a service for each pair
        # it is expected that a service will compute a confidence for *every* pair
        # that corresponds to a given identity

        for q_person_id in tqdm(self.q_person_ids,
                                desc=f"Calculating confidence matrices, provider {self.provider_id}"):

            # pandas query selecting pairs where person_ID is the name on both images
            same_person_query = f'bbox0_query_person_id=={q_person_id} and bbox1_query_person_id=={q_person_id}'  # ask for entries where the first and second person are the same and equal to the current ID
            # face_ids_for_person = self.get_ordered_face_ids_for_person(q_person_id)
            matches_for_person = self.matches_df.query(same_person_query)
            # matches_for_person = matches_for_person[(matches_for_person["bbox0_face_id"].isin(face_ids_for_person)) & (matches_for_person["bbox1_face_id"].isin(face_ids_for_person))]

            C = calculate_confidence_matrix(matches_for_person, return_full_matrix=True)
            face_ids_for_person = list(C.index)
            num_face_ids = len(face_ids_for_person)
            if not num_face_ids > 1:
                # print(f"Warning: person {q_person_id} has only {num_face_ids} faces, skipping")
                continue

            C = C.values
            self.face_ids_dict[q_person_id] = face_ids_for_person
            self.C_dict[q_person_id] = C

            C1, z, eigenvector_ratio = calculate_rank1_approximation(C, return_eigenvector_ratio=True)

            self.C1_dict[q_person_id] = C1
            self.z_dict[q_person_id] = z
            self.ev_ratios[q_person_id] = eigenvector_ratio

    def get_cropped_face(self, face_id: str):
        match = self.detections_df_long[(self.detections_df_long["face_id"] == face_id) &
                                        (self.detections_df_long["provider_id"] == self.provider_id)]
        assert len(match) == 1, f"Expected 1 match, got {len(match)}"
        match = match.iloc[0]
        photo_id = match["photo_id"]
        bbox_coords = match["bbox_coords"]
        x1, y1, x2, y2 = [int(c) for c in bbox_coords.split(",")]
        img_path = self.get_image(photo_id)
        img = Image.open(img_path).convert("RGB")
        cropped = img.crop((x1, y1, x2, y2))
        return cropped

    def get_cropped_faces_with_z_values(self, q_person_id: int):
        faces = self.face_ids_dict.get(q_person_id)
        z_values = self.z_dict.get(q_person_id)
        if faces is None or z_values is None:
            return None
        zipped = list(zip(faces, z_values))
        zipped.sort(key=lambda x: x[1])
        return [(face_id, self.get_cropped_face(face_id), z) for face_id, z in zipped]

    def get_ordered_face_ids_for_person(self, q_person_id: int):
        same_person_query = f'bbox0_query_person_id=={q_person_id} and bbox1_query_person_id=={q_person_id}'  # ask for entries where the first and second person are the same and equal to the current ID
        x = self.matches_df_full.query(same_person_query)[["bbox0_face_id", "bbox1_face_id"]]
        x["dummy"] = True
        matrix = x.drop_duplicates().pivot(index="bbox0_face_id", columns="bbox1_face_id", values="dummy").fillna(False)
        all_ids = set(matrix.index) | set(matrix.columns)
        ordered_ids = sorted(list(all_ids), key=lambda x: [int(num) for num in x.split('-')])
        matrix = matrix.reindex(index=ordered_ids, columns=ordered_ids, fill_value=False)
        matrix = matrix | matrix.T  # make symmetric

        # np.fill_diagonal(matrix.values, True)  # fill diagonal # TODO why is this failing
        for face_id in matrix.index:  # fill diagonal
            matrix.loc[face_id, face_id] = True

        return matrix[matrix.all()].index.to_list()  # return those IDs that were compared with ALL other IDs

    def display_all_confidence_matrices(self, approximate=False):

        C_dict = self.C1_dict if approximate else self.C_dict

        '''Create single display where you can see all matrices at a glance'''
        # Compute rows and cols of the multi-plot image that follows
        CC = np.floor(1.6 * np.sqrt(self.n_query_ids)).astype(int)  # columns in display
        RR = np.ceil(self.n_query_ids / CC).astype(int)  # rows in display
        fig, ax = plt.subplots(RR, CC, figsize=STANDARD_FIG_SIZE, squeeze=False)

        # disable frame for all subpanels so we don't see empty frames for unused ones
        # used ones will be activated again when filled with data
        for _c in range(CC):
            for _r in range(RR):
                ax[_r, _c].set_axis_off()

        # Display the confidence matrix: either the original and the rank1 approximation
        for i, person_ID in enumerate(self.q_person_ids):
            title = f'ID={person_ID}'
            rr, cc = np.divmod(i, CC)  # figure out the row and column of the axis to be used
            C = C_dict.get(person_ID)
            if C is not None:
                display_naked_image(C, ax[rr, cc], title)
                ax[rr, cc].set_axis_on()  # activate frame
            else:
                ax[rr, cc].set_axis_off()  # deactivate frame

        plt.suptitle(self.provider_name + " (R1 approximation)" if approximate else self.provider_name,
                     fontsize=FRONT_PAGE_FONTS)
        # self.plots.figList.append(fig)
        self.plots.save_figure(fig)


# Class to help manage the figures and output them to PDF
class PlotManager:
    '''The job of this class is collecting all the figures that are created and saving them to a PDF'''

    def __init__(self, service_meta, out_dir):
        '''Open PDF file where plots will be saved. Initialize the list of figures.'''
        self.service_meta = service_meta

        self.date_time = datetime.datetime.today().strftime('%Y-%m-%d--%H-%M')
        self.OUT_PDF_FILE_NAME = os.path.join(out_dir, 'plots/pdf_report.pdf')
        self.figList = []  # clear list of figures

    def startPDFReport(self, log: str = None):
        '''Create a title screen that displayes a few stats of the dataset.
        This will become the first page of the PDF report.'''

        LINE_SPACING = 1

        # initialize the output file pointer and the list of figures
        self.outPDF = PdfPages(self.OUT_PDF_FILE_NAME)

        # set up figure that will be used to display the opening banner
        fig = plt.figure(figsize=STANDARD_FIG_SIZE)
        plt.axis('off')

        # stuff to be printed out on the first page of the report
        if log:
            plt.text(0, 1 * LINE_SPACING, log, fontsize=FRONT_PAGE_FONTS)
        plt.text(0, 2 * LINE_SPACING, f'Included services: {", ".join(list(self.service_meta.keys()))}',
                 fontsize=FRONT_PAGE_FONTS)
        plt.text(0, 3 * LINE_SPACING, f'Date of analysis: {self.date_time}', fontsize=FRONT_PAGE_FONTS)

        plt.ylim([-1, 4 * LINE_SPACING])
        plt.gca().invert_yaxis()

        self.figList = []  # clear list of figures
        self.figList.append(fig)

    def save_figure(self, fig):
        '''Save figure to PDF file and close it'''
        self.outPDF.savefig(fig, dpi=DPI)
        plt.close(fig)

    def endPDFReport(self):
        '''endReport: Writes figures to PDF file and closes it.'''
        for fig in tqdm(self.figList, desc="Writing plots to file"):
            self.outPDF.savefig(fig, dpi=DPI)
            plt.close(fig)  # this will prevent the figures from being shown in Pyplot and will save them to PDF instead
        self.outPDF.close()
        self.figList = []  # clear list of figures


# Class to include data from all the services
class ServiceTestAll:

    def __init__(self, service_test_list, plot_mngr, person_dict, service_meta):
        '''read data from all services and store it in a table'''
        self.st_list = service_test_list
        self.plots = plot_mngr
        self.person_dict = person_dict
        self.service_meta = service_meta

        self.person_IDs = self.st_list[0].q_person_ids

    def display_all_confidence_matrices(self):
        for st in tqdm(self.st_list, desc="Creating overview pages"):
            st.plots = self.plots
            st.display_all_confidence_matrices(approximate=False)
            st.display_all_confidence_matrices(approximate=True)

    def display_individual_ID_confidences(self):
        '''Go over each individual name and display the relevant stats'''

        n_services = len(self.st_list)
        n_faces_per_row = 5 if n_services == 3 else 8

        # Display the confidence matrix: both the original and the rank1 approximation
        for i, person_ID in enumerate(tqdm(self.person_IDs, desc="Creating person ID pages")):
            plotted_anything = False

            fig = plt.figure(figsize=STANDARD_FIG_SIZE)
            outer_grid = GridSpec(4, n_services, figure=fig)

            row_0_subplots = []
            row_3_subplots = []
            for col in range(n_services):
                inner_grid = GridSpecFromSubplotSpec(1, 2, subplot_spec=outer_grid[0, col])
                inner_grid_subplots = [
                    fig.add_subplot(inner_grid[0, 0]),
                    fig.add_subplot(inner_grid[0, 1])
                ]
                row_0_subplots.append(inner_grid_subplots)

                title_ax = fig.add_subplot(inner_grid[:])
                title_ax.axis('off')
                title_ax.set_title(self.st_list[col].provider_name)

                # Face rows
                inner_grid = GridSpecFromSubplotSpec(2, n_faces_per_row, subplot_spec=outer_grid[2, col])
                inner_grid_subplots = []
                for r in [0, 1]:
                    for rr in range(n_faces_per_row):
                        ax = fig.add_subplot(inner_grid[r, rr])
                        ax.set_axis_off()  # disable frame for all subpanels so we don't see empty frames for unused ones
                        inner_grid_subplots += [ax]
                row_3_subplots.append(inner_grid_subplots)

            row_1_ax = [fig.add_subplot(outer_grid[1, i]) for i in range(n_services)]
            row_2_ax = [fig.add_subplot(outer_grid[3, i]) for i in range(n_services)]

            zs = pd.DataFrame()
            col_names = []

            for j, st in enumerate(self.st_list):
                col_names.append(self.service_meta[st.provider_name]["name"])

                face_ids = st.get_ordered_face_ids_for_person(
                    person_ID)  # get list of consensus face ids. same for all providers for a distinct person

                C = st.C_dict.get(person_ID)
                C1 = st.C1_dict.get(person_ID)
                if C is not None:
                    plotted_anything = True
                    rr, cc = C.shape
                    display_dressed_image(C, row_0_subplots[j][0], ticks=face_ids, title="Conf")
                    display_dressed_image(C1, row_0_subplots[j][1], ticks=face_ids,
                                          title=f"sv ratio={st.ev_ratios[person_ID]:.2f}")

                    hist_df = pd.DataFrame()
                    hist_df["confidence"] = C[np.triu_indices(rr, k=1)]
                    hist_df["confidence_approx_r1"] = C1[np.triu_indices(rr, k=1)]
                    sns.histplot(hist_df, bins=20, binrange=(0., 1.), kde=False, ax=row_1_ax[j], multiple="layer").set(
                        title=f'Confidence histogram {st.provider_name}')
                    zs[st.provider_id] = st.z_dict[person_ID]

                    face_crops_and_z = st.get_cropped_faces_with_z_values(person_ID)
                    if not face_crops_and_z is None:
                        # best faces
                        n_faces = min(n_faces_per_row, len(face_crops_and_z))
                        for f, (face_id, face, z) in enumerate(reversed(face_crops_and_z[-n_faces:])):
                            display_dressed_image(pad_image(face), row_3_subplots[j][f], ticks=None, vmin=None,
                                                  vmax=None,
                                                  title=f"{face_id} | z={z:.2f}", titlefontsize=5)

                        # worst faces
                        for f, (face_id, face, z) in enumerate(face_crops_and_z[:n_faces]):
                            display_dressed_image(pad_image(face), row_3_subplots[j][-(f + 1)], ticks=None, vmin=None,
                                                  vmax=None,
                                                  title=f"{face_id} | z={z:.2f}", titlefontsize=5)

            if not zs.empty:
                plotted_anything = True

                if n_services > 1:
                    service_id_to_name = {v["id"]: v["name"] for v in self.service_meta.values()}
                    zs.columns = zs.columns.map(service_id_to_name)
                    zs.index = face_ids
                    zs.plot(ax=row_2_ax[0], title="Z-values (unsorted)")
                    row_2_ax[0].set_xticks(range(len(zs)))
                    row_2_ax[0].set_xticklabels(zs.index.values, rotation=90, size=get_viewable_fontsize(len(zs)))
                    row_2_ax[0].grid(axis="x")

                if n_services == 3:
                    # TODO something seems off in this plot (x tick order)
                    plt.xticks(range(len(face_ids)), face_ids)
                    zs["mean"] = zs.mean(axis=1)
                    zs = zs.sort_values("mean").drop("mean", axis=1)
                    zs.plot(ax=row_2_ax[-2], title="Z-values (sorted by mean)")
                    row_2_ax[-2].set_xticks(range(len(zs)))
                    row_2_ax[-2].set_xticklabels(zs.index.values, rotation=90, size=get_viewable_fontsize(len(zs)))
                    row_2_ax[-2].grid(axis="x")

                plt.cla()  # clear axis
                sns.histplot(zs.reset_index(drop=True), bins=20, binrange=(0., 1.),
                             kde=False, element="poly", fill=False, ax=row_2_ax[-1]).set(title=f'Z histograms')

                plt.suptitle(f'query person id: {person_ID} ({self.person_dict[person_ID]})', fontsize=FRONT_PAGE_FONTS)

            fig.tight_layout()
            if plotted_anything:
                self.plots.save_figure(fig)
            else:
                plt.cla()
                plt.clf()
                plt.close(fig)

            # break

        return


def main(data_interface: AnalysisDataInterface, service_info: dict, out_dir: str):
    _original_stdout = sys.stdout
    if REDIRECT_STDOUT:
        stdout = sys.stdout = StringIO()

    if len(service_info) > 3:
        print("WARNING: plotting was not tested for more than 3 services.")

    os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)

    matches_df = data_interface.load_all_matches()
    detections_df_long = data_interface.load_all_detections(format="long")
    detections_df_wide = data_interface._detections_long_to_wide(detections_df_long.copy(),
                                                                 add_bbox_coords_to_wide=True)
    person_df = data_interface.get_person_meta()
    person_dict = dict(person_df[["person_id", "name"]].values)

    matches_df = join_photo_and_query_ids_to_matches(matches_df, detections_df_long)

    n_faces_detected = len(detections_df_wide)
    n_evaluations_original = len(matches_df)

    provider_counts = matches_df.groupby(["bbox0_face_id", "bbox1_face_id"])["provider_id"].count()
    n_providers_total = len(matches_df["provider_id"].unique())
    valid_pairs = provider_counts[provider_counts == n_providers_total].index
    valid_pairs = [v1 + "_" + v2 for v1, v2 in valid_pairs]
    matches_df["temp"] = matches_df["bbox0_face_id"].astype(str) + "_" + matches_df["bbox1_face_id"].astype(str)
    matches_df = matches_df[matches_df["temp"].isin(valid_pairs)]
    matches_df = matches_df.drop("temp", axis=1)
    n_evaluations_valid = len(matches_df)

    print(
        f"\nUsing {n_evaluations_valid} out of {n_evaluations_original} evaluations ({n_evaluations_valid / n_evaluations_original * 100:.2f}%).\n"
        f"The rest was discarded because not all providers had an unambiguous evaluation for the pair.\n\n")

    included_face_ids = set(matches_df["bbox0_face_id"]).union(set(matches_df["bbox1_face_id"]))

    print(
        f"Using {len(included_face_ids)} out of {n_faces_detected} faces ({len(included_face_ids) / n_faces_detected * 100:.2f}%).\n"
        f"Other face ids were not part of the pairwise evaluations.\n\n")

    st_list = []
    for provider_name, provider_data in service_info.items():
        provider_id = provider_data["id"]
        st = ServiceTest(
            provider_id=provider_id,
            provider_name=provider_name,
            matches_df=matches_df,
            detections_df_long=detections_df_long,
            image_function = data_interface.get_image_path
        )
        st_list.append(st)

    plots = PlotManager(service_info, out_dir=out_dir)
    plots.startPDFReport(log=stdout.getvalue() if REDIRECT_STDOUT else None)
    st_all = ServiceTestAll(st_list, plots, person_dict=person_dict, service_meta=service_info)
    st_all.display_all_confidence_matrices()  # TODO only execute if n_persons is not too large
    st_all.display_individual_ID_confidences()
    plots.endPDFReport()
    sys.stdout = _original_stdout