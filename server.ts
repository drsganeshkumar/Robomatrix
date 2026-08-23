import express from 'express';
import path from 'path';
import dotenv from 'dotenv';
import { GoogleGenAI, ThinkingLevel } from '@google/genai';
import { createServer as createViteServer } from 'vite';

dotenv.config();

const app = express();
const PORT = 3000;

app.use(express.json({ limit: '10mb' }));

// Lazy initialize GenAI client
let aiClient: GoogleGenAI | null = null;
function getGenAI(): GoogleGenAI {
  if (!aiClient) {
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey) {
      console.warn('GEMINI_API_KEY is not set in environment. Mock responses may be used.');
    }
    aiClient = new GoogleGenAI({
      apiKey: apiKey || 'dummy-key-for-init',
      httpOptions: {
        headers: {
          'User-Agent': 'aistudio-build',
        },
      },
    });
  }
  return aiClient;
}

// Health check endpoint
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

// System role definition for RoboMatrix AI advisor
const ROBOMATRIX_SYSTEM_INSTRUCTION = `You are RoboMatrix Agentic AI Advisor, the senior engineering and consulting assistant for RoboMatrix Autonomous Systems.
RoboMatrix is an elite software & hardware consultancy specializing in:
1. Industrial Automation & Robotics Integration (PLCs, SCADA, ROS 2, Industrial Manipulators, AMRs).
2. CEO & Founder: Dr. S. Ganeshkumar, renowned for extensive academic research, patents, and global industrial consulting expertise in Smart Manufacturing, Autonomous Mobile Robots (AMRs), and Kinematics Linkages Synthesis.
3. Next-Gen Agentic RAG for Industrial Data: Real-time telemetry ingestion, ROS 2 nodes, PLC state reasoning, SOP knowledge bases, computer vision QA inspection.
4. Automation Linkages Synthesis: Analytical & kinematic design of Toggle clamps, 4-bar linkages, Scissor lifts, Crank-slider, and Quick-return mechanisms.
5. Autonomous Robots: SLAM, Biomimetic Path Planning, Digital Twins, Multi-sensor fusion, and ISO-compliant industrial safety.

Your task is to provide knowledgeable, precise, and practical guidance to factory managers, automation engineers, robotics researchers, and industrial clients.
When answering technical questions, offer clear formulas, mechanical principles, ROS 2 / PLC integration suggestions, and actionable next steps. Invite them to schedule a detailed feasibility consultation with Dr. S. Ganeshkumar and the RoboMatrix engineering team.`;

// Multi-turn Chat endpoint
app.post('/api/gemini/chat', async (req, res) => {
  try {
    const { messages, modelChoice = 'gemini-3.7-flash', enableHighThinking = false } = req.body;

    if (!messages || !Array.isArray(messages) || messages.length === 0) {
      return res.status(400).json({ error: 'Messages array is required' });
    }

    if (!process.env.GEMINI_API_KEY) {
      // Fallback friendly simulation if key is not configured
      const lastMsg = messages[messages.length - 1]?.content || '';
      return res.json({
        reply: `[RoboMatrix AI Assistant]: Thank you for reaching out regarding "${lastMsg}". RoboMatrix specializes in autonomous robotics (AMR), ROS 2 integration, and custom kinematics synthesis guided by Dr. S. Ganeshkumar. Please attach your GEMINI_API_KEY in the Secrets panel to activate real-time neural reasoning, or submit an RFQ form below to discuss your manufacturing line directly.`,
        modelUsed: modelChoice,
      });
    }

    const ai = getGenAI();

    // Map conversation history
    const contents = messages.map((m: { role: string; content: string }) => ({
      role: m.role === 'user' ? 'user' : 'model',
      parts: [{ text: m.content }],
    }));

    // Choose model and thinking config
    let selectedModel = modelChoice;
    const config: any = {
      systemInstruction: ROBOMATRIX_SYSTEM_INSTRUCTION,
      temperature: 0.7,
    };

    if (enableHighThinking) {
      selectedModel = 'gemini-3.1-pro-preview';
      config.thinkingConfig = { thinkingLevel: ThinkingLevel.HIGH };
    }

    const response = await ai.models.generateContent({
      model: selectedModel,
      contents,
      config,
    });

    res.json({
      reply: response.text || 'No response generated.',
      modelUsed: selectedModel,
    });
  } catch (error: any) {
    console.error('Error in /api/gemini/chat:', error);
    res.status(500).json({
      error: error.message || 'Failed to process AI chat request',
    });
  }
});

// Deep Engineering Analysis / Mechanism Calculation endpoint (High Thinking Mode)
app.post('/api/gemini/analyze-system', async (req, res) => {
  try {
    const { projectType, payload, requirements } = req.body;

    if (!process.env.GEMINI_API_KEY) {
      return res.json({
        analysis: `### RoboMatrix Technical Feasibility Preview\n\n**Application:** ${projectType || 'Industrial Automation'}\n\n1. **Kinematic & Mechanism Assessment:** Evaluated based on linkage synthesis parameters. Toggle mechanism or 4-bar linkage provides optimal mechanical advantage for high clamping force.\n2. **Software & Autonomy Stack:** Recommended ROS 2 Humble/Iron with Biomimetic A* / TEB local planner for AMR navigation.\n3. **Agentic RAG Integration:** Connect PLC registers via Modbus TCP / OPC-UA to real-time vector embeddings for anomaly detection.\n\n*Configure GEMINI_API_KEY in Secrets for real-time mathematical synthesis by Dr. Ganeshkumar's AI models.*`,
        recommendations: [
          'Perform kinematic multi-body simulation in CAD/Gazebo',
          'Deploy dual LiDAR with sensor fusion for ISO 3691-4 safety compliance',
          'Implement Agentic RAG for predictive maintenance alerts',
        ],
      });
    }

    const ai = getGenAI();
    const prompt = `Perform an in-depth engineering assessment and feasibility synthesis for this industrial robotics/automation project:
Project Category: ${projectType}
User Specifications: ${JSON.stringify(payload)}
Specific Requirements: ${requirements || 'High precision, optimal mechanical advantage, low cycle time, ISO safety'}

Please output a comprehensive, structured technical breakdown including:
1. Executive Feasibility Summary
2. Mechanism & Kinematics Recommendation (e.g. 4-bar, toggle, scissor, AMR kinematics)
3. Sensor & Perception Architecture (LiDAR, Depth Camera, Multi-sensor fusion, ROS 2 topics)
4. AI & Agentic RAG Integration Strategy (Predictive analytics, SOP reasoning)
5. Expected Performance Gains (Cycle time, jam reduction, safety compliance)`;

    const response = await ai.models.generateContent({
      model: 'gemini-3.1-pro-preview',
      contents: prompt,
      config: {
        systemInstruction: ROBOMATRIX_SYSTEM_INSTRUCTION,
        thinkingConfig: {
          thinkingLevel: ThinkingLevel.HIGH,
        },
      },
    });

    res.json({
      analysis: response.text,
      modelUsed: 'gemini-3.1-pro-preview (High Thinking)',
    });
  } catch (error: any) {
    console.error('Error in /api/gemini/analyze-system:', error);
    res.status(500).json({ error: error.message || 'Analysis failed' });
  }
});

// Consultation & Google Forms submission helper
app.post('/api/consultation/submit', (req, res) => {
  const { name, email, company, projectType, requirements, timeline } = req.body;

  if (!name || !email) {
    return res.status(400).json({ error: 'Name and email are required.' });
  }

  // Pre-generate response ID & confirmation
  const consultationId = 'RMX-' + Math.floor(100000 + Math.random() * 900000);
  console.log(`[RoboMatrix] Received consultation RFQ #${consultationId} from ${name} (${company || 'Individual'}) <${email}>`);

  res.json({
    success: true,
    consultationId,
    receivedAt: new Date().toISOString(),
    message: 'Your automation inquiry has been registered. Dr. S. Ganeshkumar and the engineering team will review your specifications.',
  });
});

async function startServer() {
  if (process.env.NODE_ENV !== 'production') {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), 'dist');
    app.use(express.static(distPath));
    app.get('*', (req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`RoboMatrix server running on http://0.0.0.0:${PORT}`);
  });
}

startServer();
