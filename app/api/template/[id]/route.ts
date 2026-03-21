import { readTemplateStructureFromJson, saveTemplateStructureToJson } from "@/features/playground/libs/path-to-json";
import { db } from "@/lib/db";
import { templatePaths } from "@/lib/template";
import fs from "fs/promises";
import { NextRequest } from "next/server";
import os from "os";
import path from "path";

// Helper function to ensure valid JSON
function validateJsonStructure(data: unknown): boolean {
  try {
    JSON.parse(JSON.stringify(data)); // Ensures it's serializable
    return true;
  } catch (error) {
    console.error("Invalid JSON structure:", error);
    return false;
  }
}

async function resolveTemplatePath(templatePath: string): Promise<string | null> {
  const normalizedPath = templatePath.replace(/^\.\/?/, "");
  const candidates = [
    path.join(process.cwd(), templatePath),
    path.join(process.cwd(), normalizedPath),
    path.join(process.cwd(), "public", normalizedPath),
  ];

  for (const candidate of candidates) {
    try {
      const stats = await fs.stat(candidate);
      if (stats.isDirectory()) {
        console.log("Resolved template directory:", candidate);
        return candidate;
      }
    } catch (error) {
      // ignore not found, try next candidate
    }
  }

  return null;
}

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ id: string }> }
) {
  const param = await params;
  const id = param.id;

  if (!id) {
    return Response.json({ error: "Missing playground ID" }, { status: 400 });
  }

  const playground = await db.playground.findUnique({
    where: { id },
  });

  if (!playground) {
    return Response.json({ error: "Playground not found" }, { status: 404 });
  }

  const templateKey = playground.template as keyof typeof templatePaths;
  const templatePath = templatePaths[templateKey];

  if (!templatePath) {
    return Response.json({ error: "Invalid template" }, { status: 404 });
  }

  try {
    const resolvedInputPath = await resolveTemplatePath(templatePath);
    if (!resolvedInputPath) {
      const pathInfo = `Tried ${templatePath}, ${path.join(process.cwd(), templatePath)}, ${path.join(process.cwd(), templatePath.replace(/^\./, ""))}, ${path.join(process.cwd(), "public", templatePath.replace(/^\./, ""))}`;
      console.error("Template path resolution failed:", pathInfo);
      return Response.json({ error: "Template directory not found" }, { status: 500 });
    }

    const outputDir = path.join(os.tmpdir(), "vibecode-template-output");
    const outputFile = path.join(outputDir, `${templateKey}-${Date.now()}.json`);

    console.log("Input Path:", resolvedInputPath);
    console.log("Output Path:", outputFile);

    await fs.mkdir(outputDir, { recursive: true });

    // Save and read the template structure
    await saveTemplateStructureToJson(resolvedInputPath, outputFile);
    const result = await readTemplateStructureFromJson(outputFile);

    // Validate the JSON structure before saving
    if (!validateJsonStructure(result.items)) {
      return Response.json({ error: "Invalid JSON structure" }, { status: 500 });
    }

    await fs.unlink(outputFile);

    return Response.json({ success: true, templateJson: result }, { status: 200 });
  } catch (error) {
    console.error("Error generating template JSON:", error);
    return Response.json({ error: "Failed to generate template" }, { status: 500 });
  }
}


