from congress import Congress


class TestCongress:

    def test_tostring(self):
        values = {
            13: "13th Congress",
            14: "14th Congress",
            15: "15th Congress",
            16: "16th Congress",
            17: "17th Congress",
            18: "18th Congress",
            19: "19th Congress",
            20: "20th Congress",
            21: "21st Congress",
        }
        for (k, v) in values.items():
            congress = Congress(k)
            actual = str(congress)
            assert actual == v
