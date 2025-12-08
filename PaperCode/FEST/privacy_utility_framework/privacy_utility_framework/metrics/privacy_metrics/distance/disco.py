from typing import Any, List
import numpy as np
import pandas as pd

from privacy_utility_framework.privacy_utility_framework.metrics.privacy_metrics import PrivacyMetricCalculator
from pathlib import Path

class DisclosureCalculator(PrivacyMetricCalculator):

    def __init__(self, original: pd.DataFrame, synthetic: pd.DataFrame,
                 keys: List[str], target: str,
                 original_name: str = None, synthetic_name: str = None):
        """
        Initializes the DisclosureCalculator with datasets.

        Parameters:
            original (pd.DataFrame): Original dataset.
            synthetic (pd.DataFrame): Synthetic dataset.
            original_name (str, optional): Name for the original dataset (default: None).
            synthetic_name (str, optional): Name for the synthetic dataset (default: None).
        """
        # Initialize the superclass with datasets and settings
        super().__init__(original, synthetic,
                         original_name=original_name, synthetic_name=synthetic_name)
        self.keys = keys
        self.target = target

    def _disclosure(self, synthetic, original, keys, target):
        # Dispatch to the appropriate method based on the type of object
        if isinstance(synthetic, pd.DataFrame):
            return self._disclosure_dataframe(synthetic, original, keys, target)
        else:
            raise ValueError(f"No disclosure method associated with class {type(synthetic).__name__}")

    def _disclosure_dataframe(self, synthetic, original, keys, target, print_flag=True, compare_synorig=True):
        # Error handling for missing object
        if synthetic is None:
            raise ValueError("Requires parameter 'object' to give name of the synthetic data.")

        # Determine the number of elements, not used in our case
        if isinstance(synthetic, list) and not isinstance(synthetic, pd.DataFrame):
            m = len(synthetic)
        elif isinstance(synthetic, pd.DataFrame):
            m = 1
        else:
            raise ValueError("Object must be a data frame or a list of data frames.")

        # Adjust data using synorig.compare if needed
        if compare_synorig:
            adjust_data = self._synorig_compare(synthetic if m == 1 else synthetic[0], original, print_flag=False)
            if not adjust_data.get('unchanged', True):
                synthetic = adjust_data['syn']
                original = adjust_data['orig']
                print(
                    "Synthetic data or original or both adjusted with synorig.compare to try to make them comparable\n")
                if m > 1:
                    print("Only the first element of the list has been adjusted and will be used here\n")
                    m = 1

        # Create a custom class equivalent to R's "synds" class
        class Synds:
            def __init__(self, synds, m):
                self.synds = synds
                self.m = m

        synds_object = Synds(synds=synthetic, m=m)

        # Call the main disclosure function with the adjusted synthetic and original data
        ident, attrib = self._disclosure_synds(
            synds_object, original, keys, target=target, print_flag=print_flag,
        )

        return ident, attrib

    def _synorig_compare(self, syn, orig, print_flag=True):
        needsfix = False  # Flag to indicate if original or synthetic data needs adjustment

        unchanged = True  # Flag to indicate if data is unchanged

        if not isinstance(syn, pd.DataFrame):
            syn = pd.DataFrame(syn)
            unchanged = False
        if not isinstance(orig, pd.DataFrame):
            orig = pd.DataFrame(orig)
            unchanged = False

        # Check for variables in synthetic but not in original
        if any(col not in orig.columns for col in syn.columns):
            out = [col for col in syn.columns if col not in orig.columns]
            print(f"Variables {out} in synthetic but not in original")
            syn = syn.loc[:, syn.columns.isin(orig.columns)]
            print(f"{out} dropped from syn\n")

        # Reduce syn and orig to common vars in same order
        common = orig.columns[orig.columns.isin(syn.columns)]
        len_common = len(common)
        if print_flag:
            print(
                f"{len_common} variables in common out of {len(syn.columns)} in syn out of {len(orig.columns)} in orig")

        # Reorder to match up
        orig = orig.loc[:, orig.columns.isin(common)]
        syn = syn.loc[:, syn.columns.isin(common)]
        syn = syn.loc[:, orig.columns]

        # Change common variables that are numeric in syn and factors in orig to factors in syn
        nch_syn = 0
        nch_orig = 0
        for i in range(len_common):
            if pd.api.types.is_numeric_dtype(syn.iloc[:, i]) and isinstance(orig.iloc[:, i].dtype, pd.CategoricalDtype):
                syn.iloc[:, i] = syn.iloc[:, i].astype('category')
                nch_syn += 1
                unchanged = False
            if isinstance(syn.iloc[:, i].dtype, pd.CategoricalDtype) and pd.api.types.is_numeric_dtype(orig.iloc[:, i]):
                orig.iloc[:, i] = orig.iloc[:, i].astype('category')
                nch_orig += 1
                unchanged = False
        if not unchanged:
            print(f"\nVariables changed from numeric to factor {nch_syn} in syn {nch_orig} in original\n")

        # Change common character variables to factors
        nch_syn = 0
        nch_orig = 0
        unchanged2 = True
        for i in range(len_common):
            if pd.api.types.is_string_dtype(syn.iloc[:, i]):
                syn.iloc[:, i] = pd.Categorical(syn.iloc[:, i])
                nch_syn += 1
                unchanged2 = False
                unchanged = False
            if pd.api.types.is_string_dtype(orig.iloc[:, i]):
                orig.iloc[:, i] = pd.Categorical(orig.iloc[:, i])
                nch_orig += 1
                unchanged2 = False
                unchanged = False
        if not unchanged2:
            print(f"\nVariables changed from character to factor {nch_syn} in syn and {nch_orig} in orig\n")

        # Check data types match in common variables
        for i in range(len_common):
            if pd.api.types.is_integer_dtype(syn.iloc[:, i]) and pd.api.types.is_float_dtype(orig.iloc[:, i]):
                syn.iloc[:, i] = pd.to_numeric(syn.iloc[:, i])
                print(f"{syn.columns[i]} changed from integer to numeric in synthetic to match original")
                unchanged = False
            elif pd.api.types.is_integer_dtype(orig.iloc[:, i]) and pd.api.types.is_float_dtype(syn.iloc[:, i]):
                orig.iloc[:, i] = pd.to_numeric(orig.iloc[:, i])
                print(f"{orig.columns[i]} changed from integer to numeric in original to match synthetic")
                unchanged = False
            elif syn.iloc[:, i].dtype != orig.iloc[:, i].dtype:
                print(f'syn: {syn.iloc[:, i].dtype}')
                print(f'orig: {orig.iloc[:, i].dtype}')
                print(
                    f"\nDifferent classes for {syn.columns[i]} in syn: {syn.iloc[:, i].dtype} in orig: {orig.iloc[:, i].dtype}\n")
                needsfix = True

        # Compare missingness and levels for factors
        for i in range(len_common):
            if not orig.iloc[:, i].isna().any() and syn.iloc[:, i].isna().any():
                print(
                    f"\n\nMissing data for common variable {syn.columns[i]} in syn but not in orig\nThis looks wrong check carefully\n")
                print("PD CATEGORICAL 5")
                orig.iloc[:, i] = orig.iloc[:, i].astype('category')
                orig.iloc[:, i] = orig.iloc[:, i].fillna(value=pd.NA)

                print(f"NA added to factor {orig.columns[i]} in orig\n\n\n")
                unchanged = False
            if isinstance(syn.iloc[:, i].dtype, pd.CategoricalDtype) and isinstance(orig.iloc[:, i].dtype,
                                                                                    pd.CategoricalDtype):
                if orig.iloc[:, i].isna().any() and not syn.iloc[:, i].isna().any():
                    print("PD CATEGORICAL 6")
                    syn.iloc[:, i] = syn.iloc[:, i].astype('category').cat.add_categories([pd.NA])
                    print(f"NA added to factor {syn.columns[i]} in syn")
                    unchanged = False
                lev1 = syn.iloc[:, i].cat.categories
                lev2 = orig.iloc[:, i].cat.categories
                if len(lev1) != len(lev2) or not np.array_equal(lev1, lev2):
                    print(f"\nDifferent levels for {syn.columns[i]} in syn: {lev1} in orig: {lev2}\n")
                    all_levels = lev1.union(lev2)
                    syn.iloc[:, i] = pd.Categorical(syn.iloc[:, i], categories=all_levels, ordered=False)
                    orig.iloc[:, i] = pd.Categorical(orig.iloc[:, i], categories=all_levels, ordered=False)
                    unchanged = False
        if needsfix:
            print("\n***********************************************************************************\n"
                  "STOP: you may need to change the original or synthetic data to make them match:\n")

        if not unchanged:
            print("\n*****************************************************************\n"
                  "Differences detected and corrections attempted check output above.\n")
        else:
            print("Synthetic and original data checked with synorig.compare,\n looks like no adjustment needed\n\n")

        return {'syn': syn, 'orig': orig, 'needs_fix': needsfix, 'unchanged': unchanged}

    def _disclosure_synds(self,
                          object, data, keys, target: str, print_flag=True):

        # Check input parameters
        if not isinstance(data, pd.DataFrame):
            raise ValueError("data must be a data frame")

        data = pd.DataFrame(data)  # Ensure data is a DataFrame
        if not hasattr(object, "synds"):
            raise ValueError("object must be an object of class synds")

        if isinstance(keys, (int, float)) and not all(1 <= k <= data.shape[1] for k in keys):
            raise ValueError(f"If keys are numeric they must be in range 1 to {data.shape[1]}")

        if isinstance(keys, (int, float)):
            keys = data.columns[keys - 1]  # Adjusting for 0-based indexing in Python

        if isinstance(target, (int, float)) and not all(1 <= t <= data.shape[1] for t in target):
            raise ValueError(f"If target is numeric it must be in range 1 to {data.shape[1]}")

        if isinstance(target, (int, float)):
            target = data.columns[target - 1]  # Adjusting for 0-based indexing in Python

        if object.m == 1:
            names_syn = list(object.synds.columns)
        else:
            names_syn = list(object.synds[0].columns)

        # target must be a single variable in data and object$syn
        # keys must be a vector of variable names in data and in s
        # target must not be in keys
        if not (all(key in data.columns for key in keys) and
                all(key in names_syn for key in keys) and
                target in data.columns and
                target in names_syn):
            raise ValueError("keys and target must be variables in data and synthetic data.")

        if len(set(keys)) != len(keys):
            raise ValueError("keys cannot include duplicated values.")

        if not isinstance(target, str):
            raise ValueError("target must be a single variable.")

        if target in keys:
            raise ValueError("target cannot be in keys.")

        if 'target' in data.columns:
            raise ValueError("your data have a variable called 'target'; please rename in original and synthetic data.")

        oldkeys = keys
        keys = [col for col in data.columns if col in oldkeys]


        # Define output items
        m = object.m
        attrib = np.full((m, 8), np.nan)
        ident = np.full((m, 4), np.nan)

        attrib_cols = ["Dorig", "Dsyn", "iS", "DiS", "DiSCO", "DiSDiO", "max_denom", "mean_denom"]
        ident_cols = ["UiO", "UiS", "UiOiS", "repU"]

        attrib = pd.DataFrame(attrib, columns=attrib_cols)
        ident = pd.DataFrame(ident, columns=ident_cols)

        # Restrict data sets to targets and keys
        syndata = [object.synds] if m == 1 else object.synds
        dd = data.copy()
        dd["target"] = dd[target]

        dd = dd[["target"] + keys]
        for jj in range(m):
            syndata[jj]["target"] = syndata[jj][target]
            syndata[jj] = syndata[jj][["target"] + keys]

        # Convert remaining numeric values into factors
        numeric_vars = [col for col in dd.columns if pd.api.types.is_numeric_dtype(dd[col])]
        if numeric_vars:
            for col in numeric_vars:
                dd[col] = dd[col].astype("category")
                for j in range(object.m):
                    syndata[j][col] = syndata[j][col].astype('category')

        # Loop over each synthesis
        for jj in range(object.m):
            if print_flag:
                print(f"-------------------Synthesis {jj + 1}--------------------")

            ss = syndata[jj].copy()

            # Replace missing values with factor value of "Missing"
            def to_missing(x):
                if not isinstance(x.dtype, pd.CategoricalDtype):
                    raise ValueError(f'{x} must be a categorical type')
                x = x.astype(str)
                x[pd.isna(x)] = "Missing"
                return pd.Categorical(x)

            if dd["target"].isna().any():
                dd["target"] = to_missing(dd["target"])

            if ss["target"].isna().any():
                ss["target"] = to_missing(ss["target"])

            # Apply missing conversion to key columns
            for key in keys:
                if dd[key].isna().any():
                    dd[key] = to_missing(dd[key])

                if ss[key].isna().any():
                    ss[key] = to_missing(ss[key])

            Nd = len(dd)
            Ns = len(ss)

            # Create composite variable for keys
            if len(keys) > 1:
                ss['keys'] = ss[keys].apply(lambda row: ' | '.join(row.astype(str)), axis=1)
                dd['keys'] = dd[keys].apply(lambda row: ' | '.join(row.astype(str)), axis=1)
            else:
                ss['keys'] = ss[keys[0]]
                dd['keys'] = dd[keys[0]]

            tab_kts = pd.crosstab(ss["target"], ss['keys'])
            tab_kts.to_csv(Path(__file__).resolve().parents[5] / "examples" / "tab_kts_python.csv")
            tab_ktd = pd.crosstab(dd["target"], dd['keys'])

            if print_flag:
                print(
                    f"Table for target {target} from GT alone with keys has {tab_ktd.shape[0]} rows and {tab_ktd.shape[1]} columns.")

            # Extract unique key values
            Kd = dd['keys'].unique()
            Ks = ss['keys'].unique()

            # Extract unique target values
            Td = dd["target"].unique()
            Ts = ss["target"].unique()

            # Augment keys tables to match
            if not np.all(np.isin(Kd, Ks)):  # Some original keys not found in synthetic data
                extraKd = Kd[~np.isin(Kd, Ks)]
                extra_tab = pd.DataFrame(0, index=tab_kts.index, columns=extraKd)
                tab_kts = pd.concat([tab_kts, extra_tab], axis=1)
                tab_kts = tab_kts.reindex(sorted(tab_kts.columns), axis=1)

            if not np.all(np.isin(Ks, Kd)):  # Extra synthetic keys not in original data
                extraKs = Ks[~np.isin(Ks, Kd)]
                extra_tab = pd.DataFrame(0, index=tab_ktd.index, columns=extraKs)
                tab_ktd = pd.concat([tab_ktd, extra_tab], axis=1)
                tab_ktd = tab_ktd.reindex(sorted(tab_ktd.columns), axis=1)

            if not np.all(np.isin(Td, Ts)):  # Some original target levels not found in synthetic data
                extraTd = Td[~np.isin(Td, Ts)]
                if tab_kts.ndim == 1:
                    extra_tab = pd.DataFrame(0, index=extraTd, columns=[0])
                else:
                    extra_tab = pd.DataFrame(0, index=extraTd, columns=tab_kts.columns)
                tab_kts = pd.concat([tab_kts, extra_tab], axis=0)
                tab_kts = tab_kts.reindex(sorted(tab_kts.index), axis=0)
            else:
                extraTd = None

            if not np.all(np.isin(Ts, Td)):  # Extra synthetic target levels not in original data
                extraTs = Ts[~np.isin(Ts, Td)]
                extra_tab = pd.DataFrame(0, index=extraTs, columns=tab_ktd.columns)
                tab_ktd = pd.concat([tab_ktd, extra_tab], axis=0)
                tab_ktd = tab_ktd.reindex(sorted(tab_ktd.index), axis=0)
            else:
                extraTs = None

            if print_flag:
                print(f"Table for target {target} from GT & SD with all key combinations has "
                      f"{tab_ktd.shape[0]} rows and {tab_ktd.shape[1]} columns.")

            # Calculate proportions and margins
            tab_ktd_p = tab_ktd.div(tab_ktd.sum(axis=0), axis=1).fillna(0)
            tab_kts_p = tab_kts.div(tab_kts.sum(axis=0), axis=1).fillna(0)

            # Preparing tables for calculating attribute disclosure measures
            did = tab_ktd.copy()
            did[tab_ktd_p != 1] = 0

            dis = tab_kts.copy()
            dis[tab_kts_p != 1] = 0

            keys_syn = tab_kts.sum(axis=0)
            tab_iS = tab_ktd.copy()
            tab_iS.loc[:, keys_syn == 0] = 0

            tab_DiS = tab_ktd.copy()
            anydis = tab_kts_p.apply(lambda x: any(x == 1), axis=0)
            tab_DiS.loc[:, ~anydis] = 0

            tab_DiSCO = tab_iS.copy()
            tab_DiSCO[tab_kts_p != 1] = 0

            tab_DiSDiO = tab_DiSCO.copy()
            tab_DiSDiO[tab_ktd_p != 1] = 0


            # Identity disclosure measures calculations
            tab_ks = tab_kts.sum(axis=0)
            tab_kd = tab_ktd.sum(axis=0)

            tab_ks1 = tab_ks.copy()
            tab_ks1[tab_ks1 > 1] = 0
            tab_kd1 = tab_kd.copy()
            tab_kd1[tab_kd1 > 1] = 0
            tab_kd1_s = tab_kd1[tab_kd1.index.isin(Ks)]
            tab_ksd1 = tab_kd[(tab_ks == 1) & (tab_kd == 1)]

            UiS = tab_ks1.sum() / Ns * 100
            UiO = tab_kd1.sum() / Nd * 100
            UiOiS = tab_kd1_s.sum() / Nd * 100
            repU = tab_ksd1.sum() / Nd * 100

            # Attribute disclosure measures calculations
            Dorig = did.sum().sum() / Nd * 100
            Dsyn = dis.sum().sum() / Ns * 100
            iS = tab_iS.sum().sum() / Nd * 100
            DiS = tab_DiS.sum().sum() / Nd * 100
            DiSCO = tab_DiSCO.sum().sum() / Nd * 100
            DiSDiO = tab_DiSDiO.sum().sum() / Nd * 100
            # Ensure that jj is a valid index within the DataFrame
            if 0 <= jj < ident.shape[0]:  # Check that jj is within bounds of rows
                # Assign the calculated values to the appropriate row in the DataFrame
                ident.iloc[jj, :] = [UiO, UiS, UiOiS, repU]
            else:
                print(f"Index {jj} is out of bounds for the DataFrame with {ident.shape[0]} rows.")

            if 0 <= jj < attrib.shape[0]:  # Check that jj is within bounds of rows
                # Assign values using .iloc to target the specific row
                attrib.iloc[jj, :] = [
                    Dorig,
                    Dsyn,
                    iS,
                    DiS,
                    DiSCO,
                    DiSDiO,
                    tab_DiSCO.max().max(),
                    tab_DiSCO[tab_DiSCO > 0].mean().mean(),
                ]
            else:
                print(f"Index {jj} is out of bounds for the DataFrame with {attrib.shape[0]} rows.")
            print(f'IDENTITY: \n{ident}')
            print(f'ATTRIBUTES: \n{attrib}')
            print("~~~~~~~~~~~~~~~ Done ~~~~~~~~~~~~~~~")
            return repU, DiSCO

    def evaluate(self) -> tuple[Any, Any]:
        repU, DiSCO = self._disclosure(self.synthetic.transformed_normalized_data,
                                       self.original.transformed_normalized_data, self.keys, self.target)
        return repU, DiSCO
