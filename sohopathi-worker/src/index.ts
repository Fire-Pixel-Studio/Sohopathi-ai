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

async function askModel(question: string, studentClass: string, context: string, ollamaApiKey: string) {
  const prompt = `তুমি "সহপাঠী AI" — বাংলাদেশের ${studentClass || 'ষষ্ঠ থেকে দশম শ্রেণির'} শিক্ষার্থীদের জন্য একজন সহায়ক শিক্ষক।

নিচের তথ্য ব্যবহার করে প্রশ্নের উত্তর দাও, কিন্তু "পাঠ্যবই" বা "প্রদত্ত তথ্য" এই ধরনের শব্দ ব্যবহার কোরো না — সরাসরি বিষয়টি ব্যাখ্যা করো, যেন তুমি নিজে থেকেই জানো।

উত্তর এই নিয়ম মেনে দাও:
- প্রথমে ১-২ বাক্যে সহজ সংজ্ঞা বা মূল ধারণা বুঝিয়ে দাও (গদ্য আকারে, বুলেট পয়েন্ট নয়)
- এরপর প্রয়োজন হলে সর্বোচ্চ ২-৩টি গুরুত্বপূর্ণ পয়েন্ট বুলেট আকারে দাও — এর বেশি না
- উত্তর সংক্ষিপ্ত রাখো, অপ্রয়োজনীয় বিস্তারিত বাদ দাও
- পুরো উত্তরে সবকিছু বুলেট পয়েন্টে ভাগ করার দরকার নেই — যেখানে সাধারণ বাক্য যথেষ্ট, সেখানে বাক্যেই লেখো

ডায়াগ্রাম সম্পর্কে নিয়ম:
- যদি প্রশ্নের উত্তর কোনো গঠন, প্রক্রিয়া, বা সম্পর্ক দেখানোর মাধ্যমে বেশি স্পষ্ট হয় (যেমন: বলের দিক, কোনো বস্তুর অংশ, একটি চক্র/প্রক্রিয়ার ধাপ), তাহলে একটি সাধারণ SVG ডায়াগ্রাম যোগ করো।
- ডায়াগ্রাম শুধু তখনই দাও যখন সত্যিই প্রয়োজন — প্রতিটি উত্তরে না।
- ডায়াগ্রাম দিতে হলে এই ফরম্যাটে কোড ব্লকে দাও:
\`\`\`svg
<svg viewBox="0 0 400 220" xmlns="http://www.w3.org/2000/svg">...</svg>
\`\`\`
- শুধু সাধারণ shape ব্যবহার করো: rect, circle, line, polygon, text। কোনো script ব্যবহার কোরো না।
- রঙ: রেখা/টেক্সটের জন্য #2B2823, প্রধান অংশ বোঝাতে #C96442, দ্বিতীয় রঙ হিসেবে #2F5D3E ব্যবহার করতে পারো।
- বাংলা লেবেল অবশ্যই <text> ট্যাগে সঠিকভাবে লেখো।
- কখনোই ASCII আর্ট বা টেক্সট দিয়ে ছবি আঁকার চেষ্টা কোরো না — শুধু আসল SVG কোড দাও অথবা কিছুই দিও না।

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

      const answer = await askModel(question, studentClass, context, env.OLLAMA_API_KEY);

      return new Response(JSON.stringify({ answer }), {
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