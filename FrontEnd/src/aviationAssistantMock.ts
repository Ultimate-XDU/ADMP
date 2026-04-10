export type EvidenceLocation = {
  page: number;
  chapter: string;
  textAnchor: string;
  bbox?: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
};

export type AssistantDocumentType = "AMM" | "FIM" | "TSM" | "IPC" | "SRM";

export type AssistantCitation = {
  id: string;
  label: string;
  sourceId: string;
  documentTitle: string;
  documentType: AssistantDocumentType;
  chapter: string;
  page: number;
  quotedText: string;
  highlightText: string;
  snippet: string;
  evidenceLocation: EvidenceLocation;
};

export type AssistantSourcePage = {
  pageNumber: number;
  sectionTitle: string;
  lines: string[];
};

export type AssistantSource = {
  id: string;
  documentTitle: string;
  documentType: AssistantDocumentType;
  aircraftModel: string;
  chapter: string;
  pageCount: number;
  snippet: string;
  pages: AssistantSourcePage[];
};

export type AssistantMessageSegment = {
  id: string;
  kind: "step" | "paragraph" | "note";
  text: string;
  citationIds: string[];
};

export type AssistantMessage = {
  id: string;
  role: "user" | "assistant";
  content?: string;
  segments?: AssistantMessageSegment[];
};

export type AssistantConversation = {
  id: string;
  quickPrompt: string;
  userQuestion: string;
  aircraftModel: string;
  documentTypeFocus: AssistantDocumentType | "ALL";
  messages: AssistantMessage[];
  citations: AssistantCitation[];
  sources: AssistantSource[];
  defaultCitationId: string;
};

export const assistantDocumentTypeOptions: Array<AssistantDocumentType | "ALL"> = ["ALL", "AMM", "FIM", "TSM", "IPC", "SRM"];
export const assistantAircraftModelOptions = ["ALL", "B737-800", "A320neo", "A330-300", "C919"];

export const aviationAssistantScenarioSeeds: AssistantConversation[] = [
  {
    id: "landing-gear-hydraulic",
    quickPrompt: "如何排查起落架液压压力异常",
    userQuestion: "如何排查起落架液压压力异常？",
    aircraftModel: "B737-800",
    documentTypeFocus: "ALL",
    defaultCitationId: "lg-c1",
    messages: [
      {
        id: "lg-user",
        role: "user",
        content: "如何排查起落架液压压力异常？",
      },
      {
        id: "lg-assistant",
        role: "assistant",
        segments: [
          {
            id: "lg-seg-1",
            kind: "step",
            text: "步骤 1：先释放起落架液压供压，并确认液压泵指令已经解除，再进入后续检查支路。",
            citationIds: ["lg-c1"],
          },
          {
            id: "lg-seg-2",
            kind: "step",
            text: "步骤 2：优先检查作动筒歧管处的 O 形圈和挡圈，密封位的压扁、挤出往往是压力衰减不稳定的常见原因。",
            citationIds: ["lg-c2"],
          },
          {
            id: "lg-seg-3",
            kind: "paragraph",
            text: "如果完成密封检查后压力仍有波动，应继续按照 FIM 的隔离支路，对比泵出口压力、回油通路阻塞和作动筒泄漏情况，再沿超差支路继续排查。",
            citationIds: ["lg-c3", "lg-c4"],
          },
        ],
      },
    ],
    citations: [
      {
        id: "lg-c1",
        label: "AMM 32-11 p112",
        sourceId: "lg-s1",
        documentTitle: "Landing Gear Hydraulic Maintenance Manual",
        documentType: "AMM",
        chapter: "32-11-00",
        page: 112,
        quotedText: "Depressurize system B and confirm pump command OFF before disconnecting the gear hydraulic service line.",
        highlightText: "confirm pump command OFF before disconnecting the gear hydraulic service line",
        snippet: "Depressurize system B and confirm pump command OFF before disconnecting the gear hydraulic service line.",
        evidenceLocation: {
          page: 112,
          chapter: "32-11-00",
          textAnchor: "confirm pump command OFF before disconnecting the gear hydraulic service line",
          bbox: { x: 0.11, y: 0.31, width: 0.76, height: 0.1 },
        },
      },
      {
        id: "lg-c2",
        label: "AMM 20-22 p145",
        sourceId: "lg-s2",
        documentTitle: "Seal Replacement and Handling Practices",
        documentType: "AMM",
        chapter: "20-22-00",
        page: 145,
        quotedText: "Inspect the O-ring for flattening, nicks, extrusion, or evidence of rolling in the gland before reuse is considered.",
        highlightText: "Inspect the O-ring for flattening, nicks, extrusion, or evidence of rolling in the gland",
        snippet: "Inspect the O-ring for flattening, nicks, extrusion, or evidence of rolling in the gland before reuse is considered.",
        evidenceLocation: {
          page: 145,
          chapter: "20-22-00",
          textAnchor: "Inspect the O-ring for flattening, nicks, extrusion, or evidence of rolling in the gland",
          bbox: { x: 0.12, y: 0.46, width: 0.72, height: 0.1 },
        },
      },
      {
        id: "lg-c3",
        label: "FIM 32-31 p54",
        sourceId: "lg-s3",
        documentTitle: "Fault Isolation Manual - Landing Gear Low Pressure",
        documentType: "FIM",
        chapter: "32-31-00",
        page: 54,
        quotedText: "Compare pump outlet pressure to nominal value before isolating actuator leakage or return restriction branches.",
        highlightText: "Compare pump outlet pressure to nominal value before isolating actuator leakage or return restriction branches",
        snippet: "Compare pump outlet pressure to nominal value before isolating actuator leakage or return restriction branches.",
        evidenceLocation: {
          page: 54,
          chapter: "32-31-00",
          textAnchor: "Compare pump outlet pressure to nominal value before isolating actuator leakage or return restriction branches",
          bbox: { x: 0.1, y: 0.27, width: 0.78, height: 0.11 },
        },
      },
      {
        id: "lg-c4",
        label: "FIM 32-31 p55",
        sourceId: "lg-s3",
        documentTitle: "Fault Isolation Manual - Landing Gear Low Pressure",
        documentType: "FIM",
        chapter: "32-31-00",
        page: 55,
        quotedText: "If leakage is not detected at the actuator, inspect the return-line filter and manifold restriction path next.",
        highlightText: "inspect the return-line filter and manifold restriction path next",
        snippet: "If leakage is not detected at the actuator, inspect the return-line filter and manifold restriction path next.",
        evidenceLocation: {
          page: 55,
          chapter: "32-31-00",
          textAnchor: "inspect the return-line filter and manifold restriction path next",
          bbox: { x: 0.12, y: 0.58, width: 0.68, height: 0.1 },
        },
      },
    ],
    sources: [
      {
        id: "lg-s1",
        documentTitle: "Landing Gear Hydraulic Maintenance Manual",
        documentType: "AMM",
        aircraftModel: "B737-800",
        chapter: "32-11-00",
        pageCount: 186,
        snippet: "Depressurization sequence and hydraulic service line precautions for landing gear maintenance.",
        pages: [
          {
            pageNumber: 111,
            sectionTitle: "Preparation / Hydraulic Isolation",
            lines: [
              "1. Set the aircraft on ground maintenance condition and install all safety devices.",
              "2. Verify reservoir quantity indication is within the maintenance band.",
              "3. Open the wheel well access panel and connect the local service light.",
              "4. Remove electrical command from the landing gear hydraulic pump control circuit.",
              "5. Prepare absorbent material before any line is loosened.",
            ],
          },
          {
            pageNumber: 112,
            sectionTitle: "Depressurization / Service Line Access",
            lines: [
              "A. Depressurize system B with the approved ground maintenance sequence.",
              "B. Confirm pump command OFF before disconnecting the gear hydraulic service line.",
              "C. Install caps and plugs immediately to prevent contamination.",
              "D. Record any fluid discoloration or metallic debris for follow-up inspection.",
              "E. Continue with actuator manifold inspection only after pressure reaches zero.",
            ],
          },
        ],
      },
      {
        id: "lg-s2",
        documentTitle: "Seal Replacement and Handling Practices",
        documentType: "AMM",
        aircraftModel: "B737-800",
        chapter: "20-22-00",
        pageCount: 204,
        snippet: "Handling rules for O-rings, backup rings, and installation lubrication in hydraulic assemblies.",
        pages: [
          {
            pageNumber: 145,
            sectionTitle: "O-ring Inspection Criteria",
            lines: [
              "A. Clean the gland and inspect for scoring or embedded particles.",
              "B. Inspect the O-ring for flattening, nicks, extrusion, or evidence of rolling in the gland.",
              "C. Replace the backup ring if any edge wear or permanent set is observed.",
              "D. Apply only the approved lubricant listed in the seal replacement table.",
              "E. Do not reuse seals removed from pressurized hydraulic interfaces.",
            ],
          },
          {
            pageNumber: 146,
            sectionTitle: "Installation Notes",
            lines: [
              "A. Lubricate the seal lightly to avoid twisting during installation.",
              "B. Verify seal orientation before torque is applied to the fitting.",
              "C. Perform a low-pressure leak check before full-system repressurization.",
            ],
          },
        ],
      },
      {
        id: "lg-s3",
        documentTitle: "Fault Isolation Manual - Landing Gear Low Pressure",
        documentType: "FIM",
        aircraftModel: "B737-800",
        chapter: "32-31-00",
        pageCount: 88,
        snippet: "Troubleshooting branches for low pressure, oscillation, and leakage in landing gear hydraulic circuits.",
        pages: [
          {
            pageNumber: 54,
            sectionTitle: "Fault Isolation Entry",
            lines: [
              "1. Verify the complaint with maintenance panel pressure indication and reservoir trend.",
              "2. Compare pump outlet pressure to nominal value before isolating actuator leakage or return restriction branches.",
              "3. If pump outlet pressure is low, continue with pump command and case-drain checks.",
              "4. If pump outlet pressure is normal, continue with downstream leakage isolation.",
            ],
          },
          {
            pageNumber: 55,
            sectionTitle: "Leakage / Return Restriction Branch",
            lines: [
              "1. Inspect actuator external leakage and fitting torque first.",
              "2. If leakage is not detected at the actuator, inspect the return-line filter and manifold restriction path next.",
              "3. Replace the filter element if differential pressure exceeds the maintenance threshold.",
              "4. Repeat the pressure decay test after each corrective action.",
            ],
          },
        ],
      },
    ],
  },
  {
    id: "engine-start-failure",
    quickPrompt: "发动机启动失败的常见检查项",
    userQuestion: "发动机启动失败时应先检查哪些项目？",
    aircraftModel: "A320neo",
    documentTypeFocus: "ALL",
    defaultCitationId: "es-c1",
    messages: [
      {
        id: "es-user",
        role: "user",
        content: "发动机启动失败时应先检查哪些项目？",
      },
      {
        id: "es-assistant",
        role: "assistant",
        segments: [
          {
            id: "es-seg-1",
            kind: "step",
            text: "首先检查引气来源和启动活门通路。在排查点火硬件之前，先确认 APU 引气或交输引气压力满足手册最小要求。",
            citationIds: ["es-c1"],
          },
          {
            id: "es-seg-2",
            kind: "step",
            text: "随后核对 FADEC 启动逻辑、点火激励器可用性以及燃油关断活门指令状态。TSM 要求先隔离这三条支路，再进入附件机匣相关检查。",
            citationIds: ["es-c2", "es-c3"],
          },
          {
            id: "es-seg-3",
            kind: "note",
            text: "如果起动机已经带转但 EGT 没有上升，应优先排查燃油供给和点火输出，而不是直接拆换起动机本体。",
            citationIds: ["es-c4"],
          },
        ],
      },
    ],
    citations: [
      {
        id: "es-c1",
        label: "AMM 80-11 p27",
        sourceId: "es-s1",
        documentTitle: "Engine Start Pneumatic Supply Manual",
        documentType: "AMM",
        chapter: "80-11-00",
        page: 27,
        quotedText: "Verify available pneumatic pressure before any ignition or fuel troubleshooting begins.",
        highlightText: "Verify available pneumatic pressure before any ignition or fuel troubleshooting begins",
        snippet: "Verify available pneumatic pressure before any ignition or fuel troubleshooting begins.",
        evidenceLocation: {
          page: 27,
          chapter: "80-11-00",
          textAnchor: "Verify available pneumatic pressure before any ignition or fuel troubleshooting begins",
          bbox: { x: 0.11, y: 0.28, width: 0.72, height: 0.09 },
        },
      },
      {
        id: "es-c2",
        label: "TSM 80-12 p63",
        sourceId: "es-s2",
        documentTitle: "Troubleshooting Manual - No Light Off",
        documentType: "TSM",
        chapter: "80-12-00",
        page: 63,
        quotedText: "Check FADEC start discrete, igniter availability, and fuel shutoff valve command before mechanical teardown.",
        highlightText: "Check FADEC start discrete, igniter availability, and fuel shutoff valve command before mechanical teardown",
        snippet: "Check FADEC start discrete, igniter availability, and fuel shutoff valve command before mechanical teardown.",
        evidenceLocation: {
          page: 63,
          chapter: "80-12-00",
          textAnchor: "Check FADEC start discrete, igniter availability, and fuel shutoff valve command before mechanical teardown",
          bbox: { x: 0.12, y: 0.44, width: 0.76, height: 0.1 },
        },
      },
      {
        id: "es-c3",
        label: "FIM 80-12 p19",
        sourceId: "es-s3",
        documentTitle: "Fault Isolation Manual - Engine Start",
        documentType: "FIM",
        chapter: "80-12-00",
        page: 19,
        quotedText: "Use the no-light-off branch when starter rotation is confirmed and EGT remains unchanged.",
        highlightText: "Use the no-light-off branch when starter rotation is confirmed and EGT remains unchanged",
        snippet: "Use the no-light-off branch when starter rotation is confirmed and EGT remains unchanged.",
        evidenceLocation: {
          page: 19,
          chapter: "80-12-00",
          textAnchor: "Use the no-light-off branch when starter rotation is confirmed and EGT remains unchanged",
          bbox: { x: 0.11, y: 0.38, width: 0.74, height: 0.1 },
        },
      },
      {
        id: "es-c4",
        label: "TSM 80-12 p64",
        sourceId: "es-s2",
        documentTitle: "Troubleshooting Manual - No Light Off",
        documentType: "TSM",
        chapter: "80-12-00",
        page: 64,
        quotedText: "No EGT rise with confirmed rotation points first to fuel schedule or ignition output.",
        highlightText: "No EGT rise with confirmed rotation points first to fuel schedule or ignition output",
        snippet: "No EGT rise with confirmed rotation points first to fuel schedule or ignition output.",
        evidenceLocation: {
          page: 64,
          chapter: "80-12-00",
          textAnchor: "No EGT rise with confirmed rotation points first to fuel schedule or ignition output",
          bbox: { x: 0.13, y: 0.57, width: 0.69, height: 0.09 },
        },
      },
    ],
    sources: [
      {
        id: "es-s1",
        documentTitle: "Engine Start Pneumatic Supply Manual",
        documentType: "AMM",
        aircraftModel: "A320neo",
        chapter: "80-11-00",
        pageCount: 92,
        snippet: "Pneumatic prerequisites and minimum bleed pressure checks before engine start troubleshooting.",
        pages: [
          {
            pageNumber: 27,
            sectionTitle: "Start Air Preconditions",
            lines: [
              "A. Confirm APU bleed or cross-bleed configuration is selected as required.",
              "B. Verify available pneumatic pressure before any ignition or fuel troubleshooting begins.",
              "C. Check start valve indication and pack isolation according to the dispatch configuration.",
              "D. Record manifold pressure before initiating the troubleshooting branch.",
            ],
          },
          {
            pageNumber: 28,
            sectionTitle: "Starter Valve Verification",
            lines: [
              "A. Monitor start valve position feedback during commanded start.",
              "B. Investigate sluggish valve response before starter removal is considered.",
            ],
          },
        ],
      },
      {
        id: "es-s2",
        documentTitle: "Troubleshooting Manual - No Light Off",
        documentType: "TSM",
        aircraftModel: "A320neo",
        chapter: "80-12-00",
        pageCount: 74,
        snippet: "Engine start no-light-off logic for FADEC, ignition, and fuel command validation.",
        pages: [
          {
            pageNumber: 63,
            sectionTitle: "Logic Check / Electrical Command",
            lines: [
              "1. Confirm the starter has accelerated the engine to the expected speed.",
              "2. Check FADEC start discrete, igniter availability, and fuel shutoff valve command before mechanical teardown.",
              "3. Verify fuel metering valve motion if command is present and no light off occurs.",
            ],
          },
          {
            pageNumber: 64,
            sectionTitle: "No EGT Rise Branch",
            lines: [
              "1. No EGT rise with confirmed rotation points first to fuel schedule or ignition output.",
              "2. Swap igniter channels only if line-replaceable-unit power has already been verified.",
              "3. Continue with fuel nozzle blockage inspection after electrical checks are complete.",
            ],
          },
        ],
      },
      {
        id: "es-s3",
        documentTitle: "Fault Isolation Manual - Engine Start",
        documentType: "FIM",
        aircraftModel: "A320neo",
        chapter: "80-12-00",
        pageCount: 31,
        snippet: "Fault branches for starter rotation, no light off, slow acceleration, and start abort conditions.",
        pages: [
          {
            pageNumber: 19,
            sectionTitle: "No Light Off Entry",
            lines: [
              "1. Confirm N2 acceleration is above minimum start speed.",
              "2. Use the no-light-off branch when starter rotation is confirmed and EGT remains unchanged.",
              "3. Route to igniter, FADEC, and fuel schedule checks before hardware disassembly.",
            ],
          },
        ],
      },
    ],
  },
  {
    id: "o-ring-replacement",
    quickPrompt: "手册中关于 O 形圈更换的注意事项",
    userQuestion: "按照维修手册，更换 O 形圈时需要注意什么？",
    aircraftModel: "C919",
    documentTypeFocus: "AMM",
    defaultCitationId: "or-c1",
    messages: [
      {
        id: "or-user",
        role: "user",
        content: "按照维修手册，更换 O 形圈时需要注意什么？",
      },
      {
        id: "or-assistant",
        role: "assistant",
        segments: [
          {
            id: "or-seg-1",
            kind: "step",
            text: "清洁密封槽时只能使用批准溶剂和无纤维擦拭材料。装入新密封件之前，必须清除槽内划痕和嵌入颗粒。",
            citationIds: ["or-c1"],
          },
          {
            id: "or-seg-2",
            kind: "step",
            text: "新 O 形圈应轻度润滑，安装时避免扭转，并且从受压接头拆下的密封件不得重复使用。",
            citationIds: ["or-c2"],
          },
          {
            id: "or-seg-3",
            kind: "note",
            text: "安装完成后，在恢复系统全压之前要先进行低压检漏，尤其是液压和燃油接口位置。",
            citationIds: ["or-c3"],
          },
        ],
      },
    ],
    citations: [
      {
        id: "or-c1",
        label: "AMM 20-22 p87",
        sourceId: "or-s1",
        documentTitle: "General Seal and Gland Inspection Manual",
        documentType: "AMM",
        chapter: "20-22-00",
        page: 87,
        quotedText: "Clean the groove with approved solvent and remove all particles before seal installation.",
        highlightText: "Clean the groove with approved solvent and remove all particles before seal installation",
        snippet: "Clean the groove with approved solvent and remove all particles before seal installation.",
        evidenceLocation: {
          page: 87,
          chapter: "20-22-00",
          textAnchor: "Clean the groove with approved solvent and remove all particles before seal installation",
          bbox: { x: 0.13, y: 0.34, width: 0.67, height: 0.09 },
        },
      },
      {
        id: "or-c2",
        label: "AMM 20-22 p88",
        sourceId: "or-s1",
        documentTitle: "General Seal and Gland Inspection Manual",
        documentType: "AMM",
        chapter: "20-22-00",
        page: 88,
        quotedText: "Lubricate lightly and do not twist the seal during installation; removed seals are not to be reused.",
        highlightText: "do not twist the seal during installation; removed seals are not to be reused",
        snippet: "Lubricate lightly and do not twist the seal during installation; removed seals are not to be reused.",
        evidenceLocation: {
          page: 88,
          chapter: "20-22-00",
          textAnchor: "do not twist the seal during installation; removed seals are not to be reused",
          bbox: { x: 0.1, y: 0.51, width: 0.76, height: 0.1 },
        },
      },
      {
        id: "or-c3",
        label: "SRM 51-01 p14",
        sourceId: "or-s2",
        documentTitle: "Structural Repair and Leak Check Standard",
        documentType: "SRM",
        chapter: "51-01-00",
        page: 14,
        quotedText: "Perform a low-pressure leak check after seal replacement and before returning the component to service pressure.",
        highlightText: "Perform a low-pressure leak check after seal replacement",
        snippet: "Perform a low-pressure leak check after seal replacement and before returning the component to service pressure.",
        evidenceLocation: {
          page: 14,
          chapter: "51-01-00",
          textAnchor: "Perform a low-pressure leak check after seal replacement",
          bbox: { x: 0.12, y: 0.42, width: 0.7, height: 0.09 },
        },
      },
    ],
    sources: [
      {
        id: "or-s1",
        documentTitle: "General Seal and Gland Inspection Manual",
        documentType: "AMM",
        aircraftModel: "C919",
        chapter: "20-22-00",
        pageCount: 132,
        snippet: "Cleaning, lubrication, orientation, and reuse rules for seal replacement tasks.",
        pages: [
          {
            pageNumber: 87,
            sectionTitle: "Groove Cleaning",
            lines: [
              "A. Clean the groove with approved solvent and remove all particles before seal installation.",
              "B. Inspect the gland for scratches, burrs, and embedded metallic contamination.",
              "C. Correct any sharp edge before the new seal is unpacked.",
            ],
          },
          {
            pageNumber: 88,
            sectionTitle: "Seal Installation",
            lines: [
              "A. Lubricate the new seal lightly with approved fluid.",
              "B. Do not twist the seal during installation; removed seals are not to be reused.",
              "C. Confirm the backup ring remains seated after final assembly torque.",
            ],
          },
        ],
      },
      {
        id: "or-s2",
        documentTitle: "Structural Repair and Leak Check Standard",
        documentType: "SRM",
        aircraftModel: "C919",
        chapter: "51-01-00",
        pageCount: 41,
        snippet: "Leak check sequence after component reassembly and seal replacement.",
        pages: [
          {
            pageNumber: 14,
            sectionTitle: "Leak Check / Return To Service",
            lines: [
              "1. Perform a low-pressure leak check after seal replacement and before returning the component to service pressure.",
              "2. Hold the pressure long enough to confirm no seepage is present around the fitting.",
              "3. Record any corrective action before the access panel is closed.",
            ],
          },
        ],
      },
    ],
  },
  {
    id: "fim-fault-flow",
    quickPrompt: "查找 FIM 中某故障码的排故流程",
    userQuestion: "请查找 FIM 中某故障码对应的排故流程。",
    aircraftModel: "A330-300",
    documentTypeFocus: "FIM",
    defaultCitationId: "ff-c1",
    messages: [
      {
        id: "ff-user",
        role: "user",
        content: "请查找 FIM 中某故障码对应的排故流程。",
      },
      {
        id: "ff-assistant",
        role: "assistant",
        segments: [
          {
            id: "ff-seg-1",
            kind: "paragraph",
            text: "以故障码 LGCIU-326 为例，FIM 要求在进入任何作动筒更换支路前，先确认自检结果和插头连接完整性。",
            citationIds: ["ff-c1"],
          },
          {
            id: "ff-seg-2",
            kind: "step",
            text: "如果完成插头和供电检查后故障仍在，应继续进入接近传感器支路，并结合 TSM 中的起落架位置不一致表进行比对。",
            citationIds: ["ff-c2", "ff-c3"],
          },
          {
            id: "ff-seg-3",
            kind: "note",
            text: "维修流程明确指出，在传感器支路未排净且重复自检仍失败之前，不允许直接更换作动筒。",
            citationIds: ["ff-c4"],
          },
        ],
      },
    ],
    citations: [
      {
        id: "ff-c1",
        label: "FIM 32-42 p12",
        sourceId: "ff-s1",
        documentTitle: "Fault Isolation Manual - LGCIU Fault Codes",
        documentType: "FIM",
        chapter: "32-42-00",
        page: 12,
        quotedText: "Confirm built-in test result and connector integrity before opening any replacement branch.",
        highlightText: "Confirm built-in test result and connector integrity before opening any replacement branch",
        snippet: "Confirm built-in test result and connector integrity before opening any replacement branch.",
        evidenceLocation: {
          page: 12,
          chapter: "32-42-00",
          textAnchor: "Confirm built-in test result and connector integrity before opening any replacement branch",
          bbox: { x: 0.11, y: 0.29, width: 0.72, height: 0.1 },
        },
      },
      {
        id: "ff-c2",
        label: "FIM 32-42 p13",
        sourceId: "ff-s1",
        documentTitle: "Fault Isolation Manual - LGCIU Fault Codes",
        documentType: "FIM",
        chapter: "32-42-00",
        page: 13,
        quotedText: "Continue to the proximity sensor branch if wiring and power supply checks are satisfactory.",
        highlightText: "Continue to the proximity sensor branch if wiring and power supply checks are satisfactory",
        snippet: "Continue to the proximity sensor branch if wiring and power supply checks are satisfactory.",
        evidenceLocation: {
          page: 13,
          chapter: "32-42-00",
          textAnchor: "Continue to the proximity sensor branch if wiring and power supply checks are satisfactory",
          bbox: { x: 0.12, y: 0.48, width: 0.72, height: 0.1 },
        },
      },
      {
        id: "ff-c3",
        label: "TSM 32-42 p61",
        sourceId: "ff-s2",
        documentTitle: "Troubleshooting Manual - Gear Position Disagree",
        documentType: "TSM",
        chapter: "32-42-00",
        page: 61,
        quotedText: "Compare the returned sensor status with the gear position disagree table to isolate the failed channel.",
        highlightText: "Compare the returned sensor status with the gear position disagree table",
        snippet: "Compare the returned sensor status with the gear position disagree table to isolate the failed channel.",
        evidenceLocation: {
          page: 61,
          chapter: "32-42-00",
          textAnchor: "Compare the returned sensor status with the gear position disagree table",
          bbox: { x: 0.1, y: 0.41, width: 0.78, height: 0.09 },
        },
      },
      {
        id: "ff-c4",
        label: "FIM 32-42 p14",
        sourceId: "ff-s1",
        documentTitle: "Fault Isolation Manual - LGCIU Fault Codes",
        documentType: "FIM",
        chapter: "32-42-00",
        page: 14,
        quotedText: "Actuator replacement is not permitted until the proximity sensor branch is cleared and the repeated test remains failed.",
        highlightText: "Actuator replacement is not permitted until the proximity sensor branch is cleared",
        snippet: "Actuator replacement is not permitted until the proximity sensor branch is cleared and the repeated test remains failed.",
        evidenceLocation: {
          page: 14,
          chapter: "32-42-00",
          textAnchor: "Actuator replacement is not permitted until the proximity sensor branch is cleared",
          bbox: { x: 0.12, y: 0.56, width: 0.74, height: 0.1 },
        },
      },
    ],
    sources: [
      {
        id: "ff-s1",
        documentTitle: "Fault Isolation Manual - LGCIU Fault Codes",
        documentType: "FIM",
        aircraftModel: "A330-300",
        chapter: "32-42-00",
        pageCount: 48,
        snippet: "Fault tree for landing gear interface faults and proximity sensor disagreement.",
        pages: [
          {
            pageNumber: 12,
            sectionTitle: "Entry / Fault Confirmation",
            lines: [
              "1. Confirm built-in test result and connector integrity before opening any replacement branch.",
              "2. Inspect the connector shell, locking device, and pin seating condition.",
              "3. Verify power supply at the LGCIU before continuing.",
            ],
          },
          {
            pageNumber: 13,
            sectionTitle: "Sensor Branch",
            lines: [
              "1. Continue to the proximity sensor branch if wiring and power supply checks are satisfactory.",
              "2. Record each sensor return code before the next test cycle is initiated.",
              "3. Compare the test outcome with the disagree matrix if gear indication remains unstable.",
            ],
          },
          {
            pageNumber: 14,
            sectionTitle: "Replacement Restriction",
            lines: [
              "1. Actuator replacement is not permitted until the proximity sensor branch is cleared and the repeated test remains failed.",
              "2. Use the maintenance action log to capture every branch result before replacement approval.",
            ],
          },
        ],
      },
      {
        id: "ff-s2",
        documentTitle: "Troubleshooting Manual - Gear Position Disagree",
        documentType: "TSM",
        aircraftModel: "A330-300",
        chapter: "32-42-00",
        pageCount: 79,
        snippet: "Support table for comparing disagree status and failed sensor channels.",
        pages: [
          {
            pageNumber: 61,
            sectionTitle: "Disagree Table Mapping",
            lines: [
              "1. Compare the returned sensor status with the gear position disagree table to isolate the failed channel.",
              "2. Verify each proximity target gap before replacing the affected sensor.",
              "3. Continue with the wiring resistance branch only if the sensor gap is within tolerance.",
            ],
          },
        ],
      },
    ],
  },
];

export function cloneAssistantConversation(seed: AssistantConversation, questionOverride?: string) {
  const question = questionOverride?.trim() || seed.userQuestion;

  return {
    ...seed,
    userQuestion: question,
    messages: seed.messages.map((message) =>
      message.role === "user"
        ? { ...message, content: question }
        : {
            ...message,
            segments: message.segments?.map((segment) => ({ ...segment })),
          },
    ),
    citations: seed.citations.map((citation) => ({
      ...citation,
      evidenceLocation: {
        ...citation.evidenceLocation,
        bbox: citation.evidenceLocation.bbox ? { ...citation.evidenceLocation.bbox } : undefined,
      },
    })),
    sources: seed.sources.map((source) => ({
      ...source,
      pages: source.pages.map((page) => ({
        ...page,
        lines: [...page.lines],
      })),
    })),
  };
}

export function resolveAssistantConversation(question: string) {
  const normalized = question.trim().toLowerCase();

  const matchedSeed =
    aviationAssistantScenarioSeeds.find((seed) => {
      if (seed.id === "landing-gear-hydraulic") {
        return ["landing gear", "hydraulic", "起落架", "液压", "pressure"].some((keyword) => normalized.includes(keyword));
      }

      if (seed.id === "engine-start-failure") {
        return ["engine start", "start failure", "发动机", "点火", "light off"].some((keyword) => normalized.includes(keyword));
      }

      if (seed.id === "o-ring-replacement") {
        return ["o-ring", "seal", "密封", "replacement", "更换"].some((keyword) => normalized.includes(keyword));
      }

      return ["fault code", "fim", "故障码", "排故流程", "troubleshooting flow"].some((keyword) => normalized.includes(keyword));
    }) ?? aviationAssistantScenarioSeeds[0];

  return cloneAssistantConversation(matchedSeed, question);
}
