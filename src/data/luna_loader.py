from pathlib import Path
import SimpleITK as sitk


class LunaLoader:

    def __init__(self, dataset_path):
        self.dataset_path = Path(dataset_path)

        self.ct_path = self.dataset_path / "subset1"
        self.seg_path = self.dataset_path / "seg-lungs-LUNA16"


    def get_cases(self):

        ct_files = {
            f.stem: f for f in self.ct_path.glob("*.mhd")
        }

        seg_files = {
            f.stem: f for f in self.seg_path.glob("*.mhd")
        }

        common_cases = sorted(
            set(ct_files.keys()) &
            set(seg_files.keys())
        )

        return common_cases, ct_files, seg_files


    def load_case(self, case_id):

        _, ct_files, seg_files = self.get_cases()

        ct_image = sitk.ReadImage(
            str(ct_files[case_id])
        )

        seg_image = sitk.ReadImage(
            str(seg_files[case_id])
        )

        return ct_image, seg_image



if __name__ == "__main__":

    dataset = r"U:\OncoAegis_AI\datasets\universal_ct\luna16"

    loader = LunaLoader(dataset)

    cases, ct, seg = loader.get_cases()


    print("Matched cases:", len(cases))

    sample = cases[0]

    print("Sample ID:")
    print(sample)


    ct_img, seg_img = loader.load_case(sample)


    print("\nCT size:")
    print(ct_img.GetSize())

    print("Segmentation size:")
    print(seg_img.GetSize())


    print("\nCT spacing:")
    print(ct_img.GetSpacing())

    print("Seg spacing:")
    print(seg_img.GetSpacing())