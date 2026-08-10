import chunks from './chunks.json';

interface Env {
  OLLAMA_API_KEY: string;
  HF_API_KEY: string;
}

function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0, normA = 0, normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

async function getQueryEmbedding(text: string, hfApiKey: string) {
  const res = await fetch(
    'https://router.huggingface.co/hf-inference/models/sentence-transformers/paraphrase-multilingual-mpnet-base-v2/pipeline/feature-extraction',
    {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${hfApiKey}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ inputs: text })
    }
  );
  return res.json() as Promise<number[]>;
}

function findTopChunks(queryEmbedding: number[], topN = 2) {
  const scored = (chunks as any[]).map(chunk => ({
    chunk,
    score: cosineSimilarity(queryEmbedding, chunk.embedding)
  }));
  scored.sort((a, b) => b.score - a.score);
  return scored.slice(0, topN).map(s => s.chunk);
}

// Main text-answering model — no longer thinks about diagrams at all.
async function askModel(question: string, studentClass: string, context: string, ollamaApiKey: string) {
  const prompt = `তুমি "সহপাঠী AI" — বাংলাদেশের ${studentClass || 'ষষ্ঠ থেকে দশম শ্রেণির'} শিক্ষার্থীদের জন্য একজন সহায়ক শিক্ষক।

নিচের তথ্য ব্যবহার করে প্রশ্নের উত্তর দাও, কিন্তু "পাঠ্যবই" বা "প্রদত্ত তথ্য" এই ধরনের শব্দ ব্যবহার কোরো না — সরাসরি বিষয়টি ব্যাখ্যা করো, যেন তুমি নিজে থেকেই জানো।

উত্তর এই নিয়ম মেনে দাও:
- যদি শিক্ষার্থী শুধু একটি ছবি/ডায়াগ্রাম/চিত্র চায় (যেমন: "ছবি দাও", "ডায়াগ্রাম দেখাও"), তাহলে মাত্র ১ বাক্যে সংক্ষিপ্ত ক্যাপশন দাও — বিস্তারিত ব্যাখ্যা কোরো না।
- অন্য যেকোনো প্রশ্নের ক্ষেত্রে: প্রথমে ১-২ বাক্যে সহজ সংজ্ঞা বা মূল ধারণা বুঝিয়ে দাও (গদ্য আকারে, বুলেট পয়েন্ট নয়), এরপর প্রয়োজন হলে সর্বোচ্চ ২-৩টি গুরুত্বপূর্ণ পয়েন্ট বুলেট আকারে দাও — এর বেশি না।
- উত্তর সংক্ষিপ্ত রাখো, অপ্রয়োজনীয় বিস্তারিত বাদ দাও।
- পুরো উত্তরে সবকিছু বুলেট পয়েন্টে ভাগ করার দরকার নেই — যেখানে সাধারণ বাক্য যথেষ্ট, সেখানে বাক্যেই লেখো।
- কখনোই ASCII আর্ট বা টেক্সট দিয়ে ছবি আঁকার চেষ্টা কোরো না।

ভাষা সম্পর্কে নিয়ম:
- যদি শিক্ষার্থী প্রশ্নটি Banglish-এ লেখে (ইংরেজি হরফে বাংলা উচ্চারণ), তাহলেও প্রশ্নটি বুঝে নাও এবং স্বাভাবিক বাংলা হরফেই উত্তর দাও।

প্রাসঙ্গিক তথ্য:
${context}

প্রশ্ন: ${question}

উত্তর:`;

  const res = await fetch('https://ollama.com/api/chat', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${ollamaApiKey}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      model: 'gpt-oss:120b-cloud',
      messages: [{ role: 'user', content: prompt }],
      stream: false
    })
  });
  const data: any = await res.json();
  return data.message?.content || 'দুঃখিত, উত্তর তৈরি করা যায়নি।';
}

// Dedicated diagram model — only job is to decide if a diagram helps, and if so, produce clean SVG.
async function askDiagramModel(question: string, context: string, ollamaApiKey: string): Promise<string | null> {
  const prompt = `তুমি একজন বিজ্ঞান শিক্ষার সহায়ক ডায়াগ্রাম ডিজাইনার।

নিচের প্রশ্ন ও তথ্য দেখে সিদ্ধান্ত নাও: এই প্রশ্নের জন্য একটি সাধারণ, শিক্ষামূলক SVG ডায়াগ্রাম সত্যিই সাহায্য করবে কিনা।

যদি সাহায্য করে, শুধুমাত্র নিচের ফরম্যাটে SVG কোড দাও — অন্য কিছু লিখো না, কোনো ব্যাখ্যা দিও না:
\`\`\`svg
<svg viewBox="0 0 400 260" xmlns="http://www.w3.org/2000/svg">...</svg>
\`\`\`

নিয়ম:
- শুধু rect, circle, ellipse, line, polygon, path, text ব্যবহার করো। কোনো script, image, বা external resource ব্যবহার কোরো না।
- viewBox অন্তত 400x260 রাখো, যথেষ্ট ফাঁকা জায়গা (padding) রাখো।
- কোনো টেক্সট যেন অন্য শেপ বা অন্য টেক্সটের উপর ওভারল্যাপ না করে।
- লেবেলগুলো শেপের বাইরে রাখো, প্রয়োজনে একটি ছোট leader line দিয়ে যুক্ত করো।
- font-size কমপক্ষে 13px রাখো।
- সর্বোচ্চ ৫-৬টি মূল উপাদান/লেবেল রাখো, বেশি জটিল কোরো না।
- রঙ: রেখা/টেক্সটের জন্য #2B2823, প্রধান অংশের জন্য #C96442, দ্বিতীয় রঙ হিসেবে #2F5D3E।
- বাংলা লেবেল <text> ট্যাগে সঠিকভাবে লেখো।
- সম্পূর্ণ, বৈধ SVG কোড দাও — কখনোই অসম্পূর্ণ বা ভাঙা কোড দিও না।

যদি ডায়াগ্রাম সাহায্য না করে (যেমন সংজ্ঞা বা তারিখ জিজ্ঞাসা করা প্রশ্নে), তাহলে শুধু একটি শব্দ লেখো: NONE

প্রাসঙ্গিক তথ্য:
${context}

প্রশ্ন: ${question}`;

  const res = await fetch('https://ollama.com/api/chat', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${ollamaApiKey}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      model: 'kimi-k2.6:cloud',
      messages: [{ role: 'user', content: prompt }],
      stream: false
    })
  });
  const data: any = await res.json();
  const content: string = data.message?.content || '';

  if (content.trim() === 'NONE' || !content.includes('<svg')) return null;

  const match = content.match(/```svg\s*([\s\S]*?)```/i) || content.match(/(<svg[\s\S]*<\/svg>)/i);
  if (!match) return null;

  const svg = match[1].trim();
  // Basic sanity check — reject anything that doesn't look like real, complete SVG
  if (!svg.startsWith('<svg') || !svg.includes('</svg>')) return null;

  return svg;
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const corsHeaders = {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type'
    };

    if (request.method === 'OPTIONS') {
      return new Response(null, { headers: corsHeaders });
    }

    if (request.method !== 'POST') {
      return new Response(JSON.stringify({ error: 'Use POST' }), {
        status: 405,
        headers: { 'Content-Type': 'application/json', ...corsHeaders }
      });
    }

    try {
      const { question, studentClass } = await request.json() as { question: string; studentClass: string };

      if (!question || question.trim() === '') {
        return new Response(JSON.stringify({ error: 'প্রশ্ন খালি রাখা যাবে না' }), {
          status: 400,
          headers: { 'Content-Type': 'application/json', ...corsHeaders }
        });
      }

      const queryEmbedding = await getQueryEmbedding(question, env.HF_API_KEY);
      const topChunks = findTopChunks(queryEmbedding, 2);
      const context = topChunks.map((c: any) => c.text).join('\n\n');

      // Run the text answer and the diagram decision in parallel — same total wait time as before.
      const [answer, diagramSvg] = await Promise.all([
        askModel(question, studentClass, context, env.OLLAMA_API_KEY),
        askDiagramModel(question, context, env.OLLAMA_API_KEY)
      ]);

      const finalAnswer = diagramSvg
        ? `${answer}\n\n\`\`\`svg\n${diagramSvg}\n\`\`\``
        : answer;

      return new Response(JSON.stringify({ answer: finalAnswer }), {
        headers: { 'Content-Type': 'application/json', ...corsHeaders }
      });

    } catch (err: any) {
      return new Response(JSON.stringify({ error: err.message }), {
        status: 500,
        headers: { 'Content-Type': 'application/json', ...corsHeaders }
      });
    }
  }
};