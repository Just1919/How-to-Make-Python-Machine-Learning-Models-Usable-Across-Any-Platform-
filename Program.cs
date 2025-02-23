 using System;
using System.Net.Http;
using System.Text;
using System.Threading.Tasks;
using Newtonsoft.Json;

class Program
{
    static async Task Main(string[] args)
    {
        // Get the input data (age, sex, pclass)
        Console.Write("Enter age: ");
        var age = Console.ReadLine();

        Console.Write("Enter sex (male/female): ");
        var sex = Console.ReadLine();

        Console.Write("Enter pclass (1, 2, or 3): ");
        var pclass = Console.ReadLine();

        // Validate and ensure correct types for the input
        if (!int.TryParse(age, out int ageParsed))
        {
            Console.WriteLine("Invalid age input.");
            return;
        }

        if (!int.TryParse(pclass, out int pclassParsed) || (pclassParsed < 1 || pclassParsed > 3))
        {
            Console.WriteLine("Invalid pclass input. Please enter a value between 1 and 3.");
            return;
        }

        if (sex != "male" && sex != "female")
        {
            Console.WriteLine("Invalid sex input. Please enter 'male' or 'female'.");
            return;
        }

        // Prepare the data to send to the Flask app in JSON format
        var inputData = new
        {
            age = ageParsed,
            sex = sex,
            pclass = pclassParsed
        };

        // Convert the input data to JSON
        string jsonInput = JsonConvert.SerializeObject(inputData);

        // Create HttpClient instance to send the POST request
        var client = new HttpClient();
        var url = "http://127.0.0.1:5000/predict";  // Flask app URL

        // Send the POST request with JSON data
        var response = await client.PostAsync(url,
            new StringContent(jsonInput, Encoding.UTF8, "application/json"));

        // Check if the request was successful
        if (response.IsSuccessStatusCode)
        {
            // Read and display the response from Flask
            var responseString = await response.Content.ReadAsStringAsync();
            Console.WriteLine("Prediction Response: " + responseString);
        }
        else
        {
            Console.WriteLine("Error: Unable to get a valid response from the server.");
        }
    }
}
