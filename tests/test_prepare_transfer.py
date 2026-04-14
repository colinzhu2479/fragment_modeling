import pytest
import numpy as np
from scipy.spatial import distance as scipy_distance

from fragment_modeling.prepare_transfer import get_force_io, load_force_input

# ═════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def two_waters():
    xyz = np.array([
        -2.1522976348577143, 1.935813831410094,  1.3844868868883233,
        -2.481261367915496,  1.959085964931036,  2.293069675637073,
        -2.607829012546703,  2.6389455523580923, 0.9029163487370644,
         2.9090009231356193,-1.635249157272597,  1.1466516333609733,
         3.3888904269831794,-2.405690233783929,  0.815114885973314,
         2.999338399227558, -1.6364081663645396, 2.1084487025789698,
    ]).reshape(6, 3)
    force = np.array([
        -2.1522976348577143, 1.935813831410094,  1.3844868868883233,
        -2.481261367915496,  1.959085964931036,  2.293069675637073,
        -2.607829012546703,  2.6389455523580923, 0.9029163487370644,
         2.9090009231356193,-1.635249157272597,  1.1466516333609733,
         3.3888904269831794,-2.405690233783929,  0.815114885973314,
         2.999338399227558, -1.6364081663645396, 2.1084487025789698,
    ]).reshape(6, 3)
    atom_num  = np.array([8, 1, 1, 8, 1, 1])
    return xyz, force, atom_num


# ═════════════════════════════════════════════════════════════════════════════
# Expected outputs  (ref = np.ones([6,3]), force = xyz as above)
# ═════════════════════════════════════════════════════════════════════════════

EXPECTED = {
    'xd': [7.153480555646903, 1.552660522965384, 7.836880970649214,
           6.9830909949710875, 0.9663401245305997, 6.55733457736315,
           1.5544317427141396, 0.9660309545998327, 6.3106453346428,
           7.462861630789549, 6.579395630655631, 0.9665823368323998,
           0.9663281032586117, 7.062408028442581, 6.198854698257185],
    'xf': [-0.6944678467645101, -0.5896358920496476, -0.3305674531176864,
           -0.08801864311642332, -0.4598794879962782, -0.4290527305356251],
    'yd': [7.836880970649214, 1.552660522965384, 7.153480555646903,
           6.9830909949710875, 0.9663401245305997, 7.462861630789549,
           1.5544317427141396, 0.9663281032586117, 7.062408028442581,
           6.55733457736315, 6.579395630655631, 0.9665823368323998,
           0.9660309545998327, 6.3106453346428, 6.198854698257185],
    'yf': [0.7156744109225359, 0.9612716273900597, 2.1573441994509057,
           2.1893258941458265, 1.2328560262529307, 1.248533695310593],
    'zd': [1.5544317427141396, 7.462861630789549, 7.836880970649214,
           0.9663281032586117, 7.062408028442581, 6.55733457736315,
           7.153480555646903, 0.9660309545998327, 6.3106453346428,
           1.552660522965384, 6.579395630655631, 0.9665823368323998,
           6.9830909949710875, 0.9663401245305997, 6.198854698257185],
    'zf': [-4.123656833587842, -3.313397062753243, 3.23873980048734,
            3.6858633001195913, -3.274097283299036, 2.924660938459783],
    'e': 123.,
}


# ═════════════════════════════════════════════════════════════════════════════
# get_force_io expected value tests
# ═════════════════════════════════════════════════════════════════════════════

class TestGetForceIoExpected:

    @pytest.mark.parametrize("direction", ['x', 'y', 'z'])
    def test_dis_list_matches_expected(self, two_waters, direction):
        xyz, force, atom_num = two_waters
        dis_list, _ = get_force_io(xyz, atom_num, len(atom_num), force, direction)
        np.testing.assert_allclose(
            dis_list, EXPECTED[direction + 'd'], atol=1e-10,
            err_msg=f"dis_list mismatch for direction {direction}")

    @pytest.mark.parametrize("direction", ['x', 'y', 'z'])
    def test_force_projection_matches_expected(self, two_waters, direction):
        xyz, force, atom_num = two_waters
        _, force_t = get_force_io(xyz, atom_num, len(atom_num), force, direction)
        np.testing.assert_allclose(
            force_t, EXPECTED[direction + 'f'], atol=1e-10,
            err_msg=f"force projection mismatch for direction {direction}")


# ═════════════════════════════════════════════════════════════════════════════
# load_force_input expected value test (mocked files)
# ═════════════════════════════════════════════════════════════════════════════

class TestLoadForceInputExpected:

    def _write_two_water_files(self, tmp_path, frag_name):
        """Write files containing exactly the two-water test case."""
        xyz = np.array([
            -2.1522976348577143, 1.935813831410094,  1.3844868868883233,
            -2.481261367915496,  1.959085964931036,  2.293069675637073,
            -2.607829012546703,  2.6389455523580923, 0.9029163487370644,
             2.9090009231356193,-1.635249157272597,  1.1466516333609733,
             3.3888904269831794,-2.405690233783929,  0.815114885973314,
             2.999338399227558, -1.6364081663645396, 2.1084487025789698,
        ]).reshape(1, 6, 3)  # 1 sample, 6 atoms
        force = xyz.copy()
        atom_num = np.array([[8., 1., 1., 8., 1., 1.]])  # shape (1, 6)
        energy = np.array([123.])

        np.savetxt(tmp_path / f'{frag_name}atn.txt',          atom_num)
        np.savetxt(tmp_path / f'{frag_name}_train_xyz.txt',   xyz.reshape(1, -1))
        np.savetxt(tmp_path / f'{frag_name}_f.txt',           force.reshape(1, -1))
        np.savetxt(tmp_path / f'{frag_name}_e.txt',           energy)

    def test_full_dict_matches_expected(self, tmp_path):
        """All keys in train_dict should match EXPECTED for the two-water case."""
        self._write_two_water_files(tmp_path, 'H4O2')
        result = load_force_input('H4O2', str(tmp_path) + '/')

        for direction in ['x', 'y', 'z']:
            np.testing.assert_allclose(
                result[direction + 'd'][0], EXPECTED[direction + 'd'], atol=1e-10,
                err_msg=f"dis_list mismatch for direction {direction}")
            np.testing.assert_allclose(
                result[direction + 'f'][0], EXPECTED[direction + 'f'], atol=1e-10,
                err_msg=f"force mismatch for direction {direction}")

        assert result['e'][0] == pytest.approx(EXPECTED['e'])