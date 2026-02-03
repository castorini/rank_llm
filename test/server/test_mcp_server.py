import subprocess
import sys
import time
import unittest

import requests


class TestMCPServer(unittest.TestCase):
    def setUp(self):
        # Start the MCP server with HTTP transport
        try:
            self.server_process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "rank_llm.server.mcp",
                    "--transport",
                    "streamable-http",
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
            self.base_url = "http://127.0.0.1:8000/mcp"
            time.sleep(10)
        except Exception as e:
            self.fail(f"Failed to start MCP RankLLM server: {e}")

    def tearDown(self):
        if hasattr(self, "server_process") and self.server_process:
            self.server_process.terminate()

            try:
                # Wait up to 5 seconds for graceful termination
                self.server_process.wait(timeout=5)
                print("Process terminated gracefully")
            except subprocess.TimeoutExpired:
                print("Process didn't terminate gracefully, forcing kill...")
                # Force kill the process
                self.server_process.kill()

                try:
                    # Wait a bit more for the kill to take effect
                    self.server_process.wait(timeout=2)
                    print("Process killed forcefully")
                except subprocess.TimeoutExpired:
                    print("Warning: Process still running after kill attempt")

    def send_mcp_request(self, method, params=None):
        # Use both Content-Type and Accept headers, try both endpoints if needed
        headers = {
            "Content-Type": "application/json; charset=utf-8",
            "Accept": "application/json, text/event-stream",
        }
        # Initialize connection if needed
        if not hasattr(self, "_initialized"):
            init_request = {
                "jsonrpc": "2.0",
                "id": 0,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "test-client", "version": "1.0.0"},
                },
            }
            response = requests.post(self.base_url, json=init_request, headers=headers)
            self.assertEqual(response.status_code, 200)
            headers["mcp-session-id"] = response.headers.get("mcp-session-id")

            # Send initialized notification
            initialized_request = {
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
            }
            response = requests.post(
                self.base_url, json=initialized_request, headers=headers
            )
            self.assertTrue(response.status_code in [200, 202])
            self._initialized = True

        request = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params or {}}
        response = requests.post(self.base_url, json=request, headers=headers)
        if response.status_code != 200:
            print(f"\n==== MCP HTTP ERROR ====")
            print(f"Status: {response.status_code}")
            print(f"Response: {response.text}")
            self.fail(f"Request failed: {response.status_code} {response.text}")
        return response.text

    def test_server_starts(self):
        self.assertIsNotNone(self.server_process)
        self.assertIsNone(
            self.server_process.poll(), "Server process should be running"
        )

    def test_rerank_tool(self):
        response = self.send_mcp_request(
            "tools/call",
            {
                "name": "rerank",
                "arguments": {
                    "model_path": "Qwen/Qwen3-0.6B",
                    "query_text": "cats",
                    "candidates": [
                        {
                            "docid": "MED-2340",
                            "score": 4.735499858856201,
                            "doc": {
                                "contents": "After observing a patient allergic to cat dander and pork but devoid of other allergies, we prospectively screened patients known to be allergic to cat for a second sensitization to pork. After collecting the sera of 10 young patients found to contain specific IgE to cat dander and pork, we undertook this study to detect the possible cross-reactive allergen, define its molecular characteristics, and evaluate its clinical relevance. Through immunoblotting techniques, cat and porcine serum albumin were found to be jointly recognized molecules. These findings were further analyzed by specific anti-albumin IgE titrations and cross-inhibition experiments. Cat serum albumin cDNA was obtained from cat liver, and the corresponding amino acid sequence was deduced and compared to the known porcine and human serum albumin sequences. Inhibition experiments showed that the spectrum of IgE reactivity to cat serum albumin completely contained IgE reactivity to porcine serum albumin, suggesting that sensitization to cat was the primary event. In two cohorts of cat-allergic persons, the frequency of sensitization to cat serum albumin was found to lie between 14% and 23%. Sensitization to porcine albumin was found to lie between 3% and 10%. About 1/3 of these persons are likely to experience allergic symptoms in relation to pork consumption. Sensitization to cat serum albumin should be considered a useful marker of possible cross-sensitization not only to porcine serum albumin but also to other mammalian serum albumins."
                            },
                        },
                        {
                            "docid": "MED-3978",
                            "score": 4.263999938964844,
                            "doc": {
                                "contents": "SUMMARY The aim of this study was to investigate the relationship between dog and cat ownership and gastroenteritis in young children. A diary study of 965 children aged 4–6 years living in rural or semi-rural South Australia was undertaken. Data were collected on pet ownership, drinking water and other risk factors for gastroenteritis. Overall 89% of households had pets and dog ownership was more common than cat ownership. The multivariable models for gastroenteritis and pet ownership indicated that living in a household with a dog or cat was associated with a reduced risk of gastroenteritis (adj. OR 0·71, 95% CI 0.55–0.92; OR 0.70, % CI 0.51–0.97 respectively). This paper adds to the evidence that pets are not a major source of gastroenteritis in the home and lends support to the health benefits of pet ownership. However, this must be weighed against the potential negative consequences, such as dog bites, particularly for this age group."
                            },
                        },
                        {
                            "docid": "MED-4956",
                            "score": 4.203000068664551,
                            "doc": {
                                "contents": "Little information is available on the presence of viable Toxoplasma gondii in tissues of lambs worldwide. The prevalence of T. gondii was determined in 383 lambs (<1 year old) from Maryland, Virginia and West Virginia, USA. Hearts of 383 lambs were obtained from a slaughter house on the day of killing. Blood removed from each heart was tested for antibodies to T. gondii by using the modified agglutination test (MAT). Sera were first screened using 1:25, 1:50, 1: 100 and 1:200 dilutions, and hearts were selected for bioassay for T. gondii. Antibodies (MAT, 1:25 or higher) to T. gondii were found in 104 (27.1%) of 383 lambs. Hearts of 68 seropositive lambs were used for isolation of viable T. gondii by bioassay in cats, mice or both. For bioassays in cats, the entire myocardium or 500g was chopped and fed to cats, one cat per heart and faeces of the recipient cats were examined for shedding of T. gondii oocysts. For bioassays in mice, 50g of the myocardium was digested in an acid pepsin solution and the digest inoculated into mice; the recipient mice were examined for T. gondii infection. In total, 53 isolates of T. gondii were obtained from 68 seropositive lambs. Genotyping of the 53 T. gondii isolates using 10 PCR-restriction fragment length polymorphism markers (SAG1, SAG2, SAG3, BTUB, GRA6, c22-8, c29-2, L358, PK1 and Apico) revealed 57 strains with 15 genotypes. Four lambs had infections with two T. gondii genotypes. Twenty-six (45.6%) strains belong to the clonal Type II lineage (these strains can be further divided into two groups based on alleles at locus Apico). Eight (15.7%) strains belong to the Type III lineage. The remaining 22 strains were divided into 11 atypical genotypes. These results indicate high parasite prevalence and high genetic diversity of T. gondii in lambs, which has important implications in public health. We believe this is the first in-depth genetic analysis of T. gondii isolates from sheep in the USA."
                            },
                        },
                        {
                            "docid": "MED-3976",
                            "score": 4.135900020599365,
                            "doc": {
                                "contents": "OBJECTIVES: To investigate the effect of dog and cat contacts on the frequency of respiratory symptoms and infections during the first year of life. METHODS: In this birth cohort study, 397 children were followed up from pregnancy onward, and the frequency of respiratory symptoms and infections together with information about dog and cat contacts during the first year of life were reported by using weekly diaries and a questionnaire at the age of 1 year. All the children were born in eastern or middle Finland between September 2002 and May 2005. RESULTS: In multivariate analysis, children having dogs at home were healthier (ie, had fewer respiratory tract symptoms or infections) than children with no dog contacts (adjusted odds ratio, [aOR]: 1.31; 95% confidence interval [CI]: 1.13-1.52). Furthermore, children having dog contacts at home had less frequent otitis (aOR: 0.56; 95% CI: 0.38-0.81) and tended to need fewer courses of antibiotics (aOR: 0.71; 95% CI:0.52-0.96) than children without such contacts. In univariate analysis, both the weekly amount of contact with dogs and cats and the average yearly amount of contact were associated with decreased respiratory infectious disease morbidity. CONCLUSIONS: These results suggest that dog contacts may have a protective effect on respiratory tract infections during the first year of life. Our findings support the theory that during the first year of life, animal contacts are important, possibly leading to better resistance to infectious respiratory illnesses during childhood."
                            },
                        },
                        {
                            "docid": "MED-5060",
                            "score": 4.135899066925049,
                            "doc": {
                                "contents": "Objective To assess the association between animal exposures and non-Hodgkin lymphoma (NHL). Methods Exposure data were collected from 1,591 cases and 2,515 controls during in-person interviews in a population-based case-control study of NHL in the San Francisco Bay Area. Odds ratios (ORs) and 95% confidence intervals (CIs) were adjusted for potential confounders. Results Pet owners had a reduced risk of NHL (OR=0.71,CI=0.52 –0.97) and diffuse large-cell and immunoblastic large-cell (DLCL;OR=0.58,CI=0.39 –0.87) compared with those who never had owned a pet. Ever having owned dogs and/or cats was associated with reduced risk of all NHL (OR=0.71,CI=0.54–0.94) and of DLCL (OR=0.60,CI=0.42–0.86). Longer duration of cat ownership (p-trend=0.008), dog ownership (p-trend=0.04), and dog and/or cat ownership (p-trend =0.004) was inversely associated with risk of NHL. Ownership of pets other than cats and dogs was associated with a reduced risk of NHL (OR=0.64,CI=0.55–0.74) and DLCL (OR=0.58,CI=0.47 –0.71). Exposure to cattle for ≥5 years was associated with an increased risk of NHL (OR=1.6,CI=1.0–2.5) as was exposure to pigs for all NHL (OR=1.8,CI=1.2–2.6) and for DLCL (OR=2.0,CI=1.2–3.4). Conclusions The association between animal exposure and NHL warrants further investigation in pooled analyses."
                            },
                        },
                    ],
                    "prompt_mode": "rank_GPT",
                    "variable_passages": True,
                    "use_alpha": True,
                    "num_gpus": 1,
                    "top_k_rerank": 5,
                },
            },
        )

        self.assertTrue("result" in response)

    def test_search_tool(self):
        response = self.send_mcp_request(
            "tools/call",
            {
                "name": "search",
                "arguments": {
                    "query": "what is a lobster roll",
                    "index_name": "msmarco-v1-passage",
                    "k": 3,
                },
            },
        )

        self.assertTrue("result" in response)
