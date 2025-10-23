using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using HtmlAgilityPack;

namespace DaraGPT
{
    public class WikipediaTrainer
    {
        private readonly Config cfg;
        private readonly HashSet<string> visited = new();
        private readonly HttpClient http;

        public WikipediaTrainer(Config cfg)
        {
            this.cfg = cfg;
            http = new HttpClient();
            http.DefaultRequestHeaders.UserAgent.ParseAdd("Mozilla/5.0 (compatible; DaraGPT/1.0)");
        }

        public async Task DownloadWikipediaAsync(int maxPages)
        {
            string startUrl = "https://sh.wikipedia.org/wiki/Glavna_strana";
            Queue<string> toVisit = new();
            toVisit.Enqueue(startUrl);

            int pagesDownloaded = 0;
            string dataDir = Path.Combine(Directory.GetCurrentDirectory(), "Data");
            Directory.CreateDirectory(dataDir);

            Console.WriteLine($"Počinjem preuzimanje Wikipedia stranica (do {maxPages})...");

            while (toVisit.Count > 0 && pagesDownloaded < maxPages)
            {
                string url = toVisit.Dequeue();
                if (visited.Contains(url)) continue;
                visited.Add(url);

                Console.WriteLine($"\n[{pagesDownloaded + 1}/{maxPages}] Preuzimam: {url}");

                try
                {
                    string html = await http.GetStringAsync(url);
                    var doc = new HtmlDocument();
                    doc.LoadHtml(html);

                    // Uzmemo <p> tagove (glavni tekst)
                    var paragraphs = doc.DocumentNode.SelectNodes("//p");
                    if (paragraphs == null)
                    {
                        Console.WriteLine("Preskačem (nema paragrafa).");
                        continue;
                    }

                    string text = string.Join(" ", paragraphs.Select(p => p.InnerText));

                    // Čišćenje teksta
                    text = Regex.Replace(text, @"\[\d+\]", "");             // [1]
                    text = Regex.Replace(text, @"\s+", " ").Trim();         // višestruki razmaci
                    text = Regex.Replace(text, @"[^\wčćžšđČĆŽŠĐ\s\.,!?-]", ""); // ostavi samo slova, interpunkciju i brojke

                    if (text.Length < 150)
                    {
                        Console.WriteLine("Preskačem (tekst prekratak).");
                        continue;
                    }

                    // Kreiraj ime fajla
                    string safeName = url.Split('/').LastOrDefault() ?? $"page_{pagesDownloaded + 1}";
                    safeName = Regex.Replace(safeName, @"[^\w\-]", "_");
                    string filePath = Path.Combine(dataDir, $"{safeName}.txt");

                    await File.WriteAllTextAsync(filePath, text);
                    Console.WriteLine($"✅ Sačuvano: {filePath} ({text.Length} karaktera)");

                    pagesDownloaded++;

                    // Dodaj linkove za sledeće posete
                    var links = doc.DocumentNode.SelectNodes("//a[@href]")
                        ?.Select(a => a.GetAttributeValue("href", ""))
                        .Where(h => h.StartsWith("/wiki/") && !h.Contains(":"))
                        .Select(h => "https://sh.wikipedia.org" + h)
                        .Distinct()
                        .Take(15)
                        .ToList();

                    if (links != null)
                    {
                        int dodato = 0;
                        foreach (var link in links)
                        {
                            if (!visited.Contains(link))
                            {
                                toVisit.Enqueue(link);
                                dodato++;
                            }
                        }
                        Console.WriteLine($"Dodato {dodato} novih linkova ({toVisit.Count} u redu).");
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($" Greška pri obradi {url}: {ex.Message}");
                }
            }

            Console.WriteLine($"\nPreuzimanje završeno. Ukupno sačuvano {pagesDownloaded} stranica u folderu {dataDir}");
        }
    }
}
