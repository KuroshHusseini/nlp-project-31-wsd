import json
import os
import random

import requests

YARN_SAMPLE_URL = "https://raw.githubusercontent.com/clips/yarn/master/sample_data/abstracts_example.json"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "msh_wsd")
LOCAL_DATASET_PATH = os.path.join(DATA_DIR, "msh_wsd_dataset.json")

# Dataset based on MSH-WSD terms (Stevenson et al., 2010) with UMLS CUI labels.
# Sentences are MEDLINE-style examples for each sense.
MSH_WSD_CURATED = {
    # regular terms
    "cold": {
        "term_type": "regular",
        "senses": {
            "C0009443": {
                "label": "Common Cold",
                "sentences": [
                    "The patient presented with symptoms of a cold including runny nose and sore throat.",
                    "Rhinovirus is the most common cause of the common cold in adults.",
                    "Over-the-counter medications can help relieve cold symptoms but do not cure the infection.",
                    "She developed a cold after contact with infected individuals at work.",
                    "The cold resolved within a week without antibiotic treatment."
                ]
            },
            "C0009264": {
                "label": "Cold Temperature",
                "sentences": [
                    "Cold exposure can trigger Raynaud's phenomenon in susceptible patients.",
                    "The tissue sample was preserved under cold storage conditions at 4 degrees Celsius.",
                    "Cold therapy was applied to reduce post-operative swelling.",
                    "Hypothermia occurs when the body is exposed to extreme cold for prolonged periods.",
                    "Cold packs were applied to the injured knee to reduce inflammation."
                ]
            }
        }
    },
    "diet": {
        "term_type": "regular",
        "senses": {
            "C0012155": {
                "label": "Dietary Regimen",
                "sentences": [
                    "A low-sodium diet was prescribed to manage the patient's hypertension.",
                    "The Mediterranean diet has been shown to reduce cardiovascular disease risk.",
                    "Patients with celiac disease must maintain a strict gluten-free diet.",
                    "The ketogenic diet was evaluated as an adjunct therapy for drug-resistant epilepsy.",
                    "A high-fiber diet reduces the risk of colorectal cancer and promotes bowel health."
                ]
            },
            "C0012159": {
                "label": "Food Intake Pattern",
                "sentences": [
                    "The study assessed the habitual diet of participants using food frequency questionnaires.",
                    "Poor diet quality was associated with increased risk of metabolic syndrome.",
                    "The diet of the study population consisted primarily of processed foods.",
                    "Nutritional analysis revealed that the average diet was deficient in essential vitamins.",
                    "Improving the diet of school children was the primary goal of the intervention."
                ]
            }
        }
    },
    "discharge": {
        "term_type": "regular",
        "senses": {
            "C0030685": {
                "label": "Patient Discharge",
                "sentences": [
                    "The patient was ready for discharge after three days of inpatient treatment.",
                    "Discharge planning should begin on admission to ensure continuity of care.",
                    "Early discharge from hospital was associated with lower readmission rates in this cohort.",
                    "The discharge summary included medication reconciliation and follow-up instructions.",
                    "Premature discharge increased the risk of complications in postoperative patients."
                ]
            },
            "C0333278": {
                "label": "Bodily Discharge",
                "sentences": [
                    "Vaginal discharge was reported as a primary symptom in women with bacterial vaginosis.",
                    "Purulent discharge from the wound indicated a secondary bacterial infection.",
                    "Nipple discharge may be a sign of galactorrhea or, in rare cases, breast cancer.",
                    "Urethral discharge was observed in patients diagnosed with gonorrhea.",
                    "The character of the discharge—color, consistency, and odor—guided diagnosis."
                ]
            }
        }
    },
    "resistance": {
        "term_type": "regular",
        "senses": {
            "C0301641": {
                "label": "Vascular Resistance",
                "sentences": [
                    "Elevated pulmonary vascular resistance was identified as a complication of chronic heart failure.",
                    "Peripheral resistance determines afterload and contributes to systemic blood pressure.",
                    "Vasodilators reduce vascular resistance and improve cardiac output.",
                    "Total peripheral resistance was measured non-invasively using Doppler ultrasound.",
                    "A decrease in systemic vascular resistance was observed following administration of the drug."
                ]
            },
            "C0301642": {
                "label": "Drug Resistance",
                "sentences": [
                    "Antibiotic resistance is an increasing global health threat requiring urgent action.",
                    "Multidrug-resistant tuberculosis poses a significant challenge for treatment.",
                    "Insulin resistance is a key feature of type 2 diabetes mellitus.",
                    "Cancer cells can develop resistance to chemotherapy through various mechanisms.",
                    "The development of resistance to antimalarial drugs has complicated treatment protocols."
                ]
            }
        }
    },
    "strain": {
        "term_type": "regular",
        "senses": {
            "C0038793": {
                "label": "Bacterial Strain",
                "sentences": [
                    "The virulent strain of E. coli was isolated from contaminated food samples.",
                    "A novel strain of influenza virus was detected during the winter outbreak.",
                    "The antibiotic-resistant strain spread rapidly through the hospital ward.",
                    "Multiple strains of Staphylococcus aureus were characterized using molecular typing.",
                    "The laboratory maintained a reference strain for comparative microbiology studies."
                ]
            },
            "C0038814": {
                "label": "Muscle Strain",
                "sentences": [
                    "The athlete sustained a hamstring strain during the sprint event.",
                    "Physical therapy was recommended for the patient's lumbar strain.",
                    "Muscle strain from overexertion resulted in localized pain and tenderness.",
                    "Rest, ice, compression, and elevation were prescribed for the ankle strain.",
                    "Repetitive motion caused a strain injury to the rotator cuff tendons."
                ]
            }
        }
    },
    "mole": {
        "term_type": "regular",
        "senses": {
            "C0026919": {
                "label": "Pigmented Nevus (Skin Mole)",
                "sentences": [
                    "The dermatologist examined the mole for signs of malignant transformation.",
                    "A mole showing irregular borders and multiple colors was biopsied for histology.",
                    "Patients with numerous moles should undergo regular skin surveillance.",
                    "The mole was excised and sent for pathological examination.",
                    "Changes in the size or color of a mole may indicate early melanoma."
                ]
            },
            "C0439232": {
                "label": "Unit of Measurement (Mole)",
                "sentences": [
                    "The enzyme catalyzes the conversion of one mole of substrate per second.",
                    "Drug concentrations are often expressed in millimoles per liter.",
                    "A mole of any substance contains exactly 6.022 times ten to the power of 23 molecules.",
                    "The reaction requires two moles of oxygen for complete combustion.",
                    "Electrolyte concentrations in blood plasma are measured in millimoles per liter."
                ]
            }
        }
    },
    "pressure": {
        "term_type": "regular",
        "senses": {
            "C0005823": {
                "label": "Blood Pressure",
                "sentences": [
                    "Elevated pressure in the arterial system is a major risk factor for stroke.",
                    "The pressure was measured at both arms and found to be asymmetric.",
                    "Diastolic pressure below 60 mmHg is associated with poor coronary perfusion.",
                    "Ambulatory pressure monitoring provided a comprehensive picture of 24-hour values.",
                    "Treatment reduced the systolic pressure from 165 to 128 mmHg."
                ]
            },
            "C0033095": {
                "label": "Physical Pressure (Force)",
                "sentences": [
                    "Intracranial pressure was elevated following the traumatic brain injury.",
                    "The pressure within the eye was measured using tonometry.",
                    "Intraabdominal pressure was monitored in mechanically ventilated patients.",
                    "Atmospheric pressure changes can trigger migraine episodes in susceptible individuals.",
                    "Compartment pressure exceeding 30 mmHg indicates compartment syndrome."
                ]
            }
        }
    },
    "concentration": {
        "term_type": "regular",
        "senses": {
            "C0522510": {
                "label": "Chemical Concentration",
                "sentences": [
                    "The plasma concentration of the drug peaked at two hours after administration.",
                    "Serum concentration of glucose was measured after an overnight fast.",
                    "Therapeutic drug monitoring ensures that blood concentration remains within the target range.",
                    "The minimum inhibitory concentration was determined for each antibiotic tested.",
                    "Lead concentration in drinking water exceeded the safety threshold set by regulators."
                ]
            },
            "C1377576": {
                "label": "Cognitive Concentration",
                "sentences": [
                    "Patients with attention-deficit disorder have difficulty maintaining concentration.",
                    "Fatigue impaired the concentration and reaction time of shift workers.",
                    "Concentration and memory were assessed using standardized neuropsychological tests.",
                    "Poor concentration is a common complaint among patients with depression.",
                    "Mindfulness training improved sustained concentration in elderly participants."
                ]
            }
        }
    },
    "association": {
        "term_type": "regular",
        "senses": {
            "C0004083": {
                "label": "Statistical Association",
                "sentences": [
                    "A significant association was found between smoking and lung cancer incidence.",
                    "The association between obesity and type 2 diabetes is well established.",
                    "No statistically significant association was observed between the variables.",
                    "Genome-wide association studies have identified multiple risk loci for the disease.",
                    "The strength of the association was quantified using the odds ratio."
                ]
            },
            "C0004073": {
                "label": "Professional Association",
                "sentences": [
                    "The American Medical Association published updated clinical guidelines.",
                    "The national association of cardiologists organized an annual congress.",
                    "Membership in the association is open to all licensed practitioners.",
                    "The association released a position statement on evidence-based prescribing.",
                    "The psychiatric association endorsed the revised diagnostic criteria."
                ]
            }
        }
    },
    "secretion": {
        "term_type": "regular",
        "senses": {
            "C0036536": {
                "label": "Biological Secretion",
                "sentences": [
                    "Insulin secretion by pancreatic beta cells is impaired in type 2 diabetes.",
                    "Gastric acid secretion was reduced following proton pump inhibitor treatment.",
                    "Thyroid hormone secretion is regulated by the hypothalamic-pituitary axis.",
                    "Excessive secretion of cortisol leads to Cushing's syndrome.",
                    "The secretion of mucus protects the respiratory epithelium from pathogens."
                ]
            },
            "C0036538": {
                "label": "Secretion (Product)",
                "sentences": [
                    "Airway secretions were cleared using mechanical suction during ventilation.",
                    "Thick secretions obstructed the bronchi of patients with cystic fibrosis.",
                    "The secretion was cultured and analyzed for bacterial growth.",
                    "Purulent secretions from the ear canal suggested otitis media.",
                    "The color and viscosity of the nasal secretion helped guide antibiotic selection."
                ]
            }
        }
    },
    "weight": {
        "term_type": "regular",
        "senses": {
            "C0005910": {
                "label": "Body Weight",
                "sentences": [
                    "Patients with a body weight above 30 kg per square meter are classified as obese.",
                    "Weight loss of more than 10 percent is considered significant in oncology.",
                    "The patient's weight increased by 5 kg following initiation of corticosteroid therapy.",
                    "Low birth weight is associated with increased neonatal morbidity.",
                    "Monitoring weight weekly helps detect fluid retention in heart failure patients."
                ]
            },
            "C0439208": {
                "label": "Statistical Weight",
                "sentences": [
                    "The meta-analysis assigned weight to each study based on sample size and precision.",
                    "Weighted regression was used to account for heterogeneity across study sites.",
                    "Each observation was assigned a weight proportional to the inverse of its variance.",
                    "The weight of evidence favored the intervention over the control treatment.",
                    "Propensity score weighting was applied to balance baseline characteristics."
                ]
            }
        }
    },
    "fat": {
        "term_type": "regular",
        "senses": {
            "C0015677": {
                "label": "Dietary Fat",
                "sentences": [
                    "Saturated fat consumption was positively correlated with cardiovascular disease risk.",
                    "The diet was high in fat and low in carbohydrates as per the ketogenic protocol.",
                    "Replacing saturated fat with unsaturated fat reduced LDL cholesterol levels.",
                    "Trans fat intake was associated with increased systemic inflammation.",
                    "Dietary fat provides essential fatty acids and fat-soluble vitamins."
                ]
            },
            "C0015670": {
                "label": "Adipose Tissue (Body Fat)",
                "sentences": [
                    "Visceral fat is a strong predictor of metabolic syndrome and cardiovascular risk.",
                    "Subcutaneous fat was measured by skinfold thickness at multiple body sites.",
                    "Excessive abdominal fat is an independent risk factor for type 2 diabetes.",
                    "Brown adipose fat plays a role in thermogenesis and energy expenditure.",
                    "Fat accumulation in the liver is the hallmark of non-alcoholic fatty liver disease."
                ]
            }
        }
    },
    "frequency": {
        "term_type": "regular",
        "senses": {
            "C0376249": {
                "label": "Frequency of Occurrence",
                "sentences": [
                    "The frequency of adverse events was higher in the treatment group than placebo.",
                    "Seizure frequency decreased significantly after starting the new antiepileptic drug.",
                    "Migraine frequency was reduced from 8 to 3 episodes per month after treatment.",
                    "The frequency of hospitalizations dropped following implementation of the protocol.",
                    "Urinary frequency is a common complaint in patients with overactive bladder."
                ]
            },
            "C0016977": {
                "label": "Sound/Wave Frequency",
                "sentences": [
                    "High frequency ultrasound provides detailed images of superficial structures.",
                    "The electrical signal was filtered to retain only low frequency components.",
                    "Tinnitus was characterized as a high frequency ringing in the ears.",
                    "Nerve conduction velocity was measured at different frequencies of stimulation.",
                    "Low frequency transcranial magnetic stimulation was used to inhibit cortical excitability."
                ]
            }
        }
    },
    "operation": {
        "term_type": "regular",
        "senses": {
            "C0543467": {
                "label": "Surgical Operation",
                "sentences": [
                    "The patient underwent a complex operation to remove the tumor from the colon.",
                    "Post-operative care following the cardiac operation included intensive monitoring.",
                    "The operation was performed laparoscopically to minimize recovery time.",
                    "Intraoperative complications extended the duration of the operation by two hours.",
                    "Written informed consent was obtained before the operation was performed."
                ]
            },
            "C0677878": {
                "label": "Operational Process",
                "sentences": [
                    "The laboratory operation followed strict quality control procedures.",
                    "The operation of the medical device was evaluated for safety and efficacy.",
                    "Standard operation procedures were updated following the audit recommendations.",
                    "The operation of the healthcare system was analyzed using process mapping.",
                    "Proper operation of the ventilator requires specialized training."
                ]
            }
        }
    },
    "support": {
        "term_type": "regular",
        "senses": {
            "C0175780": {
                "label": "Emotional or Social Support",
                "sentences": [
                    "Family support was identified as a key factor in recovery from substance use disorder.",
                    "Psychosocial support improved quality of life in patients with terminal illness.",
                    "Peer support groups helped patients manage chronic pain more effectively.",
                    "Social support was associated with lower rates of depression in elderly patients.",
                    "Caregiver support programs reduced burnout among family members of dementia patients."
                ]
            },
            "C0037713": {
                "label": "Physical or Mechanical Support",
                "sentences": [
                    "Ventilatory support was initiated due to respiratory failure.",
                    "External cardiac support devices were used in patients with cardiogenic shock.",
                    "Orthotic support reduced pain and improved function in patients with flat feet.",
                    "Nutritional support via nasogastric tube was provided during the acute phase.",
                    "Life support was withdrawn following a determination of brain death."
                ]
            }
        }
    },

    # acronyms
    "ms": {
        "term_type": "acronym",
        "senses": {
            "C0026769": {
                "label": "Multiple Sclerosis",
                "sentences": [
                    "Patients diagnosed with MS often experience relapsing and remitting disease courses.",
                    "MRI of the brain showed multiple demyelinating lesions consistent with MS.",
                    "Disease-modifying therapies can slow the progression of MS.",
                    "Fatigue is one of the most disabling symptoms reported by patients with MS.",
                    "The diagnosis of MS was confirmed using the McDonald diagnostic criteria."
                ]
            },
            "C0026826": {
                "label": "Mitral Stenosis",
                "sentences": [
                    "Rheumatic fever is the most common cause of acquired MS in developing countries.",
                    "Echocardiography confirmed severe MS with a mitral valve area of 0.9 cm squared.",
                    "Percutaneous balloon mitral valvuloplasty was performed for symptomatic MS.",
                    "The murmur characteristic of MS was heard best at the apex with the bell.",
                    "MS leads to elevated left atrial pressure and pulmonary congestion."
                ]
            },
            "C0026906": {
                "label": "Mass Spectrometry",
                "sentences": [
                    "Liquid chromatography MS was used to quantify trace amounts of the drug.",
                    "Tandem MS enabled identification of phosphopeptides in proteomic studies.",
                    "Metabolomics studies rely on high-resolution MS to profile small molecules.",
                    "MS confirmed the molecular weight of the synthesized compound.",
                    "Gas chromatography coupled with MS identified volatile organic compounds in exhaled air."
                ]
            }
        }
    },
    "bp": {
        "term_type": "acronym",
        "senses": {
            "C0005823": {
                "label": "Blood Pressure",
                "sentences": [
                    "Ambulatory BP monitoring over 24 hours revealed a nocturnal dipping pattern.",
                    "Target BP for hypertensive patients with diabetes is below 130 over 80 mmHg.",
                    "Elevated BP was recorded at three consecutive clinic visits before treatment was started.",
                    "Self-monitoring of BP at home improved adherence to antihypertensive medication.",
                    "BP was measured after five minutes of rest in the seated position."
                ]
            },
            "C0174727": {
                "label": "British Pharmacopoeia",
                "sentences": [
                    "Drug purity was assessed according to BP specifications and reference standards.",
                    "The BP monograph was used to verify the identity of the active pharmaceutical ingredient.",
                    "Compliance with BP standards is mandatory for medicines licensed in the United Kingdom.",
                    "Dissolution testing was performed in accordance with the BP apparatus method.",
                    "The formulation met all quality criteria as defined in the current edition of the BP."
                ]
            }
        }
    },
    "pe": {
        "term_type": "acronym",
        "senses": {
            "C0034067": {
                "label": "Pulmonary Embolism",
                "sentences": [
                    "CT pulmonary angiography confirmed bilateral PE in the patient.",
                    "Anticoagulation with low molecular weight heparin was initiated for acute PE.",
                    "D-dimer elevation raised the clinical suspicion of PE in this postoperative patient.",
                    "Massive PE causing haemodynamic instability was treated with systemic thrombolysis.",
                    "The Wells score was calculated to estimate the pre-test probability of PE."
                ]
            },
            "C1947907": {
                "label": "Physical Examination",
                "sentences": [
                    "PE revealed splenomegaly and tender hepatomegaly consistent with portal hypertension.",
                    "Findings on PE were unremarkable except for mild ankle edema.",
                    "Cardiovascular PE showed a regular rate and rhythm with no murmurs.",
                    "The patient consented to a full PE including neurological assessment.",
                    "Abdominal PE detected a pulsatile mass suggesting aortic aneurysm."
                ]
            }
        }
    },
    "dm": {
        "term_type": "acronym",
        "senses": {
            "C0011849": {
                "label": "Diabetes Mellitus",
                "sentences": [
                    "Patients with DM require regular screening for diabetic retinopathy.",
                    "HbA1c levels above 6.5 percent confirm the diagnosis of DM.",
                    "Type 2 DM is associated with insulin resistance and progressive beta cell failure.",
                    "Metformin remains the first-line oral agent for newly diagnosed type 2 DM.",
                    "Foot care education is essential to prevent ulceration in patients with DM."
                ]
            },
            "C0013240": {
                "label": "Dermatomyositis",
                "sentences": [
                    "DM is an inflammatory myopathy characterized by skin rash and proximal muscle weakness.",
                    "Heliotrope rash around the eyelids is a classic dermatological feature of DM.",
                    "Gottron papules overlying the metacarpophalangeal joints suggest DM.",
                    "Muscle biopsy in DM shows perifascicular atrophy and inflammatory infiltrate.",
                    "Anti-Jo-1 antibodies are associated with interstitial lung disease complicating DM."
                ]
            }
        }
    },
    "cf": {
        "term_type": "acronym",
        "senses": {
            "C0010674": {
                "label": "Cystic Fibrosis",
                "sentences": [
                    "CFTR mutations cause CF, a life-limiting autosomal recessive disorder.",
                    "Regular physiotherapy to clear secretions is essential in the management of CF.",
                    "Pancreatic enzyme replacement therapy is required by most patients with CF.",
                    "Lung transplantation may be considered for patients with end-stage CF.",
                    "Newborn screening programs allow early diagnosis and treatment of CF."
                ]
            },
            "C0007785": {
                "label": "Cerebrospinal Fluid",
                "sentences": [
                    "Analysis of CF revealed elevated protein and lymphocytosis consistent with viral meningitis.",
                    "CF pressure was measured at 28 cm of water confirming raised intracranial pressure.",
                    "Oligoclonal bands in the CF support the diagnosis of multiple sclerosis.",
                    "Blood was detected in the CF following subarachnoid hemorrhage.",
                    "Glucose levels in the CF were below serum levels, suggesting bacterial infection."
                ]
            }
        }
    },
    "ac": {
        "term_type": "acronym",
        "senses": {
            "C0001786": {
                "label": "Acupuncture",
                "sentences": [
                    "AC was used as an adjunct therapy for chronic low back pain management.",
                    "A randomized trial evaluated the efficacy of AC versus sham treatment for headache.",
                    "Patients receiving AC reported greater improvement in pain scores than controls.",
                    "Traditional Chinese medicine advocates AC for a wide range of conditions.",
                    "The mechanisms underlying AC analgesia are still under investigation."
                ]
            },
            "C0220611": {
                "label": "Adenylyl Cyclase",
                "sentences": [
                    "Beta receptor stimulation activates AC and increases intracellular cyclic AMP.",
                    "Inhibition of AC reduced cAMP production and attenuated the inflammatory response.",
                    "AC isoforms show tissue-specific expression patterns in the cardiovascular system.",
                    "G-protein coupled receptor signaling acts through AC to modulate cellular function.",
                    "Cholera toxin permanently activates AC leading to massive fluid secretion."
                ]
            }
        }
    },
    "cp": {
        "term_type": "acronym",
        "senses": {
            "C0007789": {
                "label": "Cerebral Palsy",
                "sentences": [
                    "CP is a group of non-progressive disorders affecting motor function.",
                    "Spastic CP is the most common type and results from cortical or subcortical injury.",
                    "Early physiotherapy intervention improves functional outcomes in children with CP.",
                    "Botulinum toxin injections reduce spasticity in selected children with CP.",
                    "The Gross Motor Function Classification System is used to grade severity in CP."
                ]
            },
            "C0008036": {
                "label": "Chest Pain",
                "sentences": [
                    "The patient presented with acute CP radiating to the left arm.",
                    "Non-cardiac causes of CP include esophageal spasm and musculoskeletal disorders.",
                    "Troponin measurement is essential in the evaluation of acute CP.",
                    "Unstable angina manifests as rest CP without ST-elevation on ECG.",
                    "CP on exertion that resolves with rest is the hallmark of stable angina."
                ]
            }
        }
    },
    "gp": {
        "term_type": "acronym",
        "senses": {
            "C0017368": {
                "label": "General Practitioner",
                "sentences": [
                    "The GP referred the patient to a cardiologist for further evaluation.",
                    "Primary care is delivered by a GP who acts as the first point of contact.",
                    "The GP prescribed antihypertensive medication and arranged follow-up in six weeks.",
                    "Regular GP visits allowed early detection of the patient's hypertension.",
                    "The GP coordinated care between multiple specialists to manage the chronic conditions."
                ]
            },
            "C0017382": {
                "label": "Glycoprotein",
                "sentences": [
                    "Platelet GP IIb/IIIa inhibitors are used to prevent thrombosis during coronary interventions.",
                    "The viral envelope GP mediates cell entry and is a target for vaccine development.",
                    "Surface GP expression was quantified by flow cytometry.",
                    "P-selectin is a GP expressed on activated platelets and endothelial cells.",
                    "GP Ib is the receptor for von Willebrand factor and is essential for platelet adhesion."
                ]
            }
        }
    },
    "fa": {
        "term_type": "acronym",
        "senses": {
            "C0015684": {
                "label": "Fatty Acid",
                "sentences": [
                    "Omega-3 FA supplementation reduced triglyceride levels in dyslipidemic patients.",
                    "Free FA released from adipose tissue provide an energy substrate during fasting.",
                    "Essential FA cannot be synthesized de novo and must be obtained from the diet.",
                    "Plasma FA profiling revealed elevated saturated FA in obese patients.",
                    "Short-chain FA produced by gut bacteria have anti-inflammatory properties."
                ]
            },
            "C0015695": {
                "label": "Folic Acid",
                "sentences": [
                    "FA supplementation before and during pregnancy prevents neural tube defects.",
                    "Deficiency of FA leads to megaloblastic anemia with hypersegmented neutrophils.",
                    "Patients on methotrexate therapy require concurrent FA to reduce toxicity.",
                    "Serum FA levels were measured to evaluate nutritional status in elderly patients.",
                    "Fortification of cereal products with FA has reduced the incidence of spina bifida."
                ]
            }
        }
    },
    "mr": {
        "term_type": "acronym",
        "senses": {
            "C0024485": {
                "label": "Magnetic Resonance Imaging",
                "sentences": [
                    "Brain MR demonstrated multiple cortical and subcortical lesions.",
                    "Cardiac MR quantified ventricular volumes and ejection fraction precisely.",
                    "MR spectroscopy provided metabolic information about the lesion.",
                    "Diffusion-weighted MR imaging was used to identify acute ischemic stroke.",
                    "Whole-body MR was performed to assess disease extent in multiple myeloma."
                ]
            },
            "C0026266": {
                "label": "Mitral Regurgitation",
                "sentences": [
                    "Echocardiography revealed severe MR with a broad eccentric regurgitant jet.",
                    "Surgical repair is preferred over replacement for MR caused by leaflet prolapse.",
                    "Chronic MR causes left ventricular volume overload and eventually dilatation.",
                    "Transcatheter edge-to-edge repair is an emerging treatment for inoperable MR.",
                    "MR severity was graded using quantitative echocardiographic parameters."
                ]
            }
        }
    }
}


def build_dataset_from_curated():
    """Convert the curated MSH-WSD dict into evaluation-ready format."""
    dataset = {}
    for term, info in MSH_WSD_CURATED.items():
        senses = info["senses"]
        dataset[term] = {
            "term_type": info["term_type"],
            "instances": []
        }
        for cui, sense_info in senses.items():
            for sentence in sense_info["sentences"]:
                dataset[term]["instances"].append({
                    "sentence": sentence,
                    "cui": cui,
                    "cui_label": sense_info["label"]
                })
    return dataset


def fetch_yarn_sample():
    """Fetch the yarn sample data from GitHub."""
    try:
        r = requests.get(YARN_SAMPLE_URL, timeout=15)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {}


def load_or_create_dataset():
    """Load MSH-WSD dataset from disk or build it from the curated definitions."""
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(LOCAL_DATASET_PATH):
        with open(LOCAL_DATASET_PATH) as f:
            return json.load(f)

    # Build and save
    dataset = build_dataset_from_curated()

    # Also try to include yarn sample data (converted to compatible format)
    yarn_sample = fetch_yarn_sample()
    for term, senses_by_cui in yarn_sample.items():
        t_lower = term.lower()
        if t_lower in dataset:
            continue
        # Determine if acronym by checking length and capitalization
        term_type = "acronym" if len(term) <= 4 and term.upper() == term else "regular"
        dataset[t_lower] = {
            "term_type": term_type,
            "instances": []
        }
        for cui, sentences in senses_by_cui.items():
            for sent in sentences:
                dataset[t_lower]["instances"].append({
                    "sentence": sent,
                    "cui": cui,
                    "cui_label": cui  # label is just the CUI when no name is known
                })

    with open(LOCAL_DATASET_PATH, "w") as f:
        json.dump(dataset, f, indent=2)

    return dataset


def get_term_types(dataset):
    """Return (regular_terms, acronyms) lists."""
    regular = [t for t, info in dataset.items() if info["term_type"] == "regular"]
    acronyms = [t for t, info in dataset.items() if info["term_type"] == "acronym"]
    return regular, acronyms


def sample_terms(dataset, n_regular=10, n_acronyms=10, seed=42):
    """Randomly sample terms for evaluation using a fixed seed."""
    random.seed(seed)
    regular, acronyms = get_term_types(dataset)
    selected_regular = random.sample(regular, min(n_regular, len(regular)))
    selected_acronyms = random.sample(acronyms, min(n_acronyms, len(acronyms)))
    return selected_regular, selected_acronyms


def get_dataset_stats(dataset):
    """Return summary statistics about the loaded dataset."""
    regular, acronyms = get_term_types(dataset)
    stats = {
        "total_terms": len(dataset),
        "regular_terms": len(regular),
        "acronyms": len(acronyms),
        "total_instances": sum(len(info["instances"]) for info in dataset.values()),
        "avg_instances_per_term": round(
            sum(len(info["instances"]) for info in dataset.values()) / len(dataset), 1
        ) if dataset else 0,
    }
    return stats


if __name__ == "__main__":
    dataset = load_or_create_dataset()
    stats = get_dataset_stats(dataset)
    print("Dataset Statistics:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    regular, acronyms = sample_terms(dataset)
    print(f"\nSampled regular terms: {regular}")
    print(f"Sampled acronyms: {acronyms}")
